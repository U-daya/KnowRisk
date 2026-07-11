import { useRef, useEffect, useState, useMemo } from 'react';
import Globe from 'react-globe.gl';
import * as THREE from 'three';
import type { ComponentRiskDetail } from './api';

// lat, lng centroids for countries appearing in the dataset.
const COUNTRY_COORDS: Record<string, [number, number]> = {
  'Taiwan':      [23.7, 121.0],
  'USA':         [39.8, -98.5],
  'China':       [35.9, 104.2],
  'Japan':       [36.2, 138.3],
  'South Korea': [36.5, 127.8],
  'Netherlands': [52.1, 5.3],
  'Germany':     [51.2, 10.5],
  'Israel':      [31.0, 34.8],
  'Malaysia':    [4.2, 101.9],
};

function riskColor(score: number): string {
  if (score >= 0.5) return '#ef4444'; // critical/high - red
  if (score >= 0.2) return '#f59e0b'; // medium - amber
  return '#22c55e'; // low - green
}

interface GlobeMapProps {
  detail: ComponentRiskDetail;
}

interface LabelPart {
  name: string;
  color: string;
}

interface LabelPoint {
  lat: number;
  lng: number;
  title: string;
  parts: LabelPart[];
  color: string;
  isDest: boolean;
}

// Altitude (as a fraction of globe radius) that labels float above the surface.
const LABEL_ALTITUDE = 0.04;

export function GlobeMap({ detail }: GlobeMapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const globeRef = useRef<any>(null);
  const [size, setSize] = useState({ width: 800, height: 600 });

  useEffect(() => {
    if (!containerRef.current) return;
    const el = containerRef.current;
    const observer = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setSize({ width, height });
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const destCoord = COUNTRY_COORDS[detail.country];

  // Group dependencies by origin country so arcs from the same country can
  // fan out (different altitudes) instead of stacking on top of each other.
  const arcs = useMemo(() => {
    if (!destCoord) return [];
    const byCountry = new Map<string, number>();
    return detail.dependency_risks
      .filter((dep) => COUNTRY_COORDS[dep.country])
      .map((dep) => {
        const [lat, lng] = COUNTRY_COORDS[dep.country];
        const idx = byCountry.get(dep.country) ?? 0;
        byCountry.set(dep.country, idx + 1);
        return {
          startLat: lat,
          startLng: lng,
          endLat: destCoord[0],
          endLng: destCoord[1],
          color: riskColor(dep.risk_score),
          partName: dep.name,
          originCountry: dep.country,
          riskScore: dep.risk_score,
          altitude: 0.25 + idx * 0.08,
        };
      });
  }, [detail, destCoord]);

  // Labels — one per unique origin country, listing EVERY part shipped from
  // there as its own line (colored by that part's own risk level), plus one
  // label for the destination component itself.
  const labels: LabelPoint[] = useMemo(() => {
    if (!destCoord) return [];
    const byCountry = new Map<string, LabelPart[]>();
    for (const dep of detail.dependency_risks) {
      if (!COUNTRY_COORDS[dep.country]) continue;
      const list = byCountry.get(dep.country) ?? [];
      list.push({ name: dep.name, color: riskColor(dep.risk_score) });
      byCountry.set(dep.country, list);
    }

    // Parts whose origin country is the SAME as the destination component's
    // own country would land on the exact same lat/lng as the destination
    // label and render on top of it (garbled overlapping text). Pull those
    // out and fold them into the destination label instead of giving them
    // their own overlapping label.
    const sameCountryParts = byCountry.get(detail.country) ?? [];
    byCountry.delete(detail.country);

    const originLabels: LabelPoint[] = Array.from(byCountry.entries()).map(([country, parts]) => {
      const [lat, lng] = COUNTRY_COORDS[country];
      return { lat, lng, title: country, parts, color: '#ffffff', isDest: false };
    });

    return [
      ...originLabels,
      {
        lat: destCoord[0],
        lng: destCoord[1],
        title: detail.component_name,
        parts: [
          { name: detail.country, color: '#ffffff' },
          ...sameCountryParts,
        ],
        color: '#ffffff',
        isDest: true,
      },
    ];
  }, [detail, destCoord]);

  const points = useMemo(() => {
    if (!destCoord) return [];
    return [
      { lat: destCoord[0], lng: destCoord[1], name: detail.component_name, size: 1.4, color: '#ffffff' },
      ...arcs.map((a) => ({ lat: a.startLat, lng: a.startLng, name: a.partName, size: 0.7, color: a.color })),
    ];
  }, [arcs, destCoord, detail.component_name]);

  useEffect(() => {
    if (globeRef.current && destCoord) {
      globeRef.current.pointOfView({ lat: destCoord[0], lng: destCoord[1], altitude: 2.2 }, 1000);
      const controls = globeRef.current.controls();
      if (controls) {
        controls.autoRotate = true;
        controls.autoRotateSpeed = 0.4;
      }
    }
  }, [detail, destCoord]);

  if (!destCoord) {
    return (
      <div className="flex-1 flex items-center justify-center text-zinc-500 text-sm uppercase tracking-wide">
        No coordinates available for {detail.country}.
      </div>
    );
  }

  return (
    <div ref={containerRef} className="flex-1 min-h-0 relative overflow-hidden">
      <Globe
        ref={globeRef}
        width={size.width}
        height={size.height}
        backgroundColor="rgba(0,0,0,0)"
        globeImageUrl="//unpkg.com/three-globe/example/img/earth-dark.jpg"
        bumpImageUrl="//unpkg.com/three-globe/example/img/earth-topology.png"
        // Arcs
        arcsData={arcs}
        arcColor="color"
        arcAltitude="altitude"
        arcDashLength={0.4}
        arcDashGap={0.2}
        arcDashAnimateTime={1500}
        arcStroke={0.6}
        arcLabel={(d: any) => `${d.partName} — ${d.originCountry} → ${detail.country} (risk ${d.riskScore.toFixed(2)})`}
        // Small colored dots at each origin/destination
        pointsData={points}
        pointLat="lat"
        pointLng="lng"
        pointColor="color"
        pointRadius="size"
        pointAltitude={0.01}
      />
      {/*
        Always-on text labels, positioned manually every frame from the
        globe's own camera + projection matrix — instead of three-globe's
        `htmlElementsData` layer. That layer writes its own `transform` onto
        the label's root element every frame to place it on screen; any of
        our own styling on that same root fights it, and the label can end
        up positioned at (0,0) or off-canvas with no visible error. This
        overlay owns 100% of its own positioning, so it can't silently break
        that way again.
      */}
      <GlobeLabelOverlay globeRef={globeRef} labels={labels} size={size} />
    </div>
  );
}

function GlobeLabelOverlay({
  globeRef,
  labels,
  size,
}: {
  globeRef: React.RefObject<any>;
  labels: LabelPoint[];
  size: { width: number; height: number };
}) {
  const elRefs = useRef<(HTMLDivElement | null)[]>([]);
  const rafRef = useRef<number>();
  const warnedRef = useRef(false);

  useEffect(() => {
    const tick = () => {
      const g = globeRef.current;
      const hasCamera = !!g && typeof g.camera === 'function';
      const hasGetCoords = !!g && typeof g.getCoords === 'function';

      if (!warnedRef.current && g && (!hasCamera || !hasGetCoords)) {
        warnedRef.current = true;
        // eslint-disable-next-line no-console
        console.error(
          'GlobeLabelOverlay: globe ref is missing expected methods.',
          { hasCamera, hasGetCoords, globeRefKeys: Object.keys(g) }
        );
      }

      if (g && hasCamera && hasGetCoords) {
        const camera = g.camera();
        if (!warnedRef.current && !camera) {
          warnedRef.current = true;
          // eslint-disable-next-line no-console
          console.error('GlobeLabelOverlay: g.camera() returned falsy', camera);
        }
        if (camera) {
          // Direction from the globe's center (0,0,0) out to the camera —
          // used to tell which side of the sphere is currently facing us.
          const camDir = camera.position.clone().normalize();

          labels.forEach((label, i) => {
            const el = elRefs.current[i];
            if (!el) return;

            const c = g.getCoords(label.lat, label.lng, LABEL_ALTITUDE);
            const vec = new THREE.Vector3(c.x, c.y, c.z);

            // How directly this point faces the camera: 1 = dead center of
            // the visible hemisphere, 0 = exactly on the horizon, negative =
            // around on the far side (should be hidden).
            const facing = vec.clone().normalize().dot(camDir);
            const opacity = Math.max(0, Math.min(1, (facing - 0.05) / 0.2));

            if (opacity <= 0.01) {
              el.style.display = 'none';
              return;
            }

            const projected = vec.clone().project(camera);
            const x = ((projected.x + 1) / 2) * size.width;
            const y = ((1 - projected.y) / 2) * size.height;

            el.style.display = 'block';
            el.style.transform = `translate(${x}px, ${y}px) translate(-50%, -100%)`;
            el.style.opacity = String(opacity);
          });
        }
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [globeRef, labels, size]);

  return (
    <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none', overflow: 'hidden', zIndex: 20 }}>
      {labels.map((label, i) => (
        <div
          key={`${label.title}-${i}`}
          ref={(node) => (elRefs.current[i] = node)}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            pointerEvents: 'none',
            fontFamily: 'monospace',
            whiteSpace: 'nowrap',
            textAlign: 'center',
            lineHeight: 1.35,
          }}
        >
          <div
            style={{
              fontSize: 11,
              fontWeight: label.isDest ? 700 : 600,
              color: label.color,
              textShadow: '0 0 4px #000, 0 0 4px #000',
            }}
          >
            {label.title}
          </div>
          {label.parts.map((part, j) => (
            <div
              key={j}
              style={{
                fontSize: 9,
                color: part.color,
                textShadow: '0 0 4px #000, 0 0 4px #000',
              }}
            >
              {part.name}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}