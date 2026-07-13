import { useRef, useEffect, useState, useMemo } from 'react';
import Globe from 'react-globe.gl';
import * as THREE from 'three';
import type { ComponentRiskDetail, MergedComponent } from './api';

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
  // Full component graph (each node carries its own `dependencies`), used to
  // trace the second hop (Tier 2 -> Tier 3) beyond the selected component's
  // direct dependencies.
  components: MergedComponent[];
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

interface Arc {
  startLat: number;
  startLng: number;
  endLat: number;
  endLng: number;
  color: string;
  partName: string;
  originCountry: string;
  endCountry: string;
  riskScore: number;
  altitude: number;
  tier: 2 | 3; // 2 = direct dependency arc, 3 = upstream (dependency-of-dependency)
}

interface Pt {
  lat: number;
  lng: number;
  name: string;
  size: number;
  color: string;
}

// Altitude (as a fraction of globe radius) that labels float above the surface.
const LABEL_ALTITUDE = 0.04;

export function GlobeMap({ detail, components }: GlobeMapProps) {
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

  const compById = useMemo(
    () => new Map(components.map((c) => [c.id, c])),
    [components],
  );

  // Tier 1<-2 arcs: each DIRECT dependency's country -> the selected part's
  // country. Fan out by origin country so same-country arcs don't overlap.
  const arcs2 = useMemo<Arc[]>(() => {
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
          endCountry: detail.country,
          riskScore: dep.risk_score,
          altitude: 0.25 + idx * 0.08,
          tier: 2 as const,
        };
      });
  }, [detail, destCoord]);

  // Tier 2<-3 arcs: for every direct dependency, trace ITS dependencies one
  // more hop and draw a fainter, lower arc from the upstream part's country to
  // the direct dependency's country. This is the third layer of the chain.
  const arcs3 = useMemo<Arc[]>(() => {
    if (!destCoord) return [];
    const seen = new Set<string>();
    const byPair = new Map<string, number>();
    const out: Arc[] = [];
    for (const dep of detail.dependency_risks) {
      const depCoord = COUNTRY_COORDS[dep.country];
      if (!depCoord) continue;
      const depNode = compById.get(dep.id);
      if (!depNode) continue;
      for (const subId of depNode.dependencies) {
        const sub = compById.get(subId);
        if (!sub) continue;
        const subCoord = COUNTRY_COORDS[sub.country];
        if (!subCoord) continue;
        const key = `${sub.id}->${dep.id}`;
        if (seen.has(key)) continue;
        seen.add(key);
        const pairKey = `${sub.country}->${dep.country}`;
        const idx = byPair.get(pairKey) ?? 0;
        byPair.set(pairKey, idx + 1);
        out.push({
          startLat: subCoord[0],
          startLng: subCoord[1],
          endLat: depCoord[0],
          endLng: depCoord[1],
          color: riskColor(sub.risk_score),
          partName: sub.name,
          originCountry: sub.country,
          endCountry: dep.country,
          riskScore: sub.risk_score,
          altitude: 0.1 + idx * 0.05,
          tier: 3 as const,
        });
      }
    }
    return out;
  }, [detail, destCoord, compById]);

  const allArcs = useMemo<Arc[]>(() => [...arcs2, ...arcs3], [arcs2, arcs3]);

  // Labels — one per unique origin country (across BOTH tiers), listing every
  // part shipped from there, plus one label for the destination component.
  const labels: LabelPoint[] = useMemo(() => {
    if (!destCoord) return [];
    const byCountry = new Map<string, LabelPart[]>();
    const push = (country: string, part: LabelPart) => {
      if (!COUNTRY_COORDS[country]) return;
      const list = byCountry.get(country) ?? [];
      if (!list.some((p) => p.name === part.name)) list.push(part);
      byCountry.set(country, list);
    };
    for (const dep of detail.dependency_risks) {
      push(dep.country, { name: dep.name, color: riskColor(dep.risk_score) });
    }
    for (const a of arcs3) {
      push(a.originCountry, { name: a.partName, color: riskColor(a.riskScore) });
    }

    // Parts whose origin country is the destination's own country would land
    // on top of the destination label; fold them in instead.
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
        parts: [{ name: detail.country, color: '#ffffff' }, ...sameCountryParts],
        color: '#ffffff',
        isDest: true,
      },
    ];
  }, [detail, destCoord, arcs3]);

  const points = useMemo<Pt[]>(() => {
    if (!destCoord) return [];
    const pts: Pt[] = [
      { lat: destCoord[0], lng: destCoord[1], name: detail.component_name, size: 1.4, color: '#ffffff' },
    ];
    for (const a of allArcs) {
      pts.push({
        lat: a.startLat,
        lng: a.startLng,
        name: a.partName,
        size: a.tier === 3 ? 0.45 : 0.7,
        color: a.color,
      });
    }
    return pts;
  }, [allArcs, destCoord, detail.component_name]);

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
        // Arcs — tier 3 rendered thinner, lower, and with a longer dash cycle
        // so the two layers read as distinct depths.
        arcsData={allArcs}
        arcColor="color"
        arcAltitude="altitude"
        arcDashLength={(d: any) => (d.tier === 3 ? 0.25 : 0.4)}
        arcDashGap={(d: any) => (d.tier === 3 ? 0.3 : 0.2)}
        arcDashAnimateTime={(d: any) => (d.tier === 3 ? 2600 : 1500)}
        arcStroke={(d: any) => (d.tier === 3 ? 0.28 : 0.6)}
        arcLabel={(d: any) =>
          `${d.partName} — ${d.originCountry} → ${d.endCountry} (Tier ${d.tier}, risk ${d.riskScore.toFixed(2)})`
        }
        // Small colored dots at each origin/destination
        pointsData={points}
        pointLat="lat"
        pointLng="lng"
        pointColor="color"
        pointRadius="size"
        pointAltitude={0.01}
      />

      {/* Tier legend */}
      <div
        style={{ position: 'absolute', left: 12, bottom: 12, zIndex: 20, pointerEvents: 'none' }}
        className="font-mono text-[9px] uppercase tracking-wide leading-relaxed text-zinc-400"
      >
        <div>
          <span style={{ color: '#e4e4e7' }}>━━</span> Tier 1 ← 2 · direct supply
        </div>
        <div>
          <span style={{ color: '#71717a' }}>┄┄</span> Tier 2 ← 3 · upstream supply
        </div>
      </div>

      {/*
        Always-on text labels, positioned manually every frame from the
        globe's own camera + projection matrix (see original note): this
        overlay owns 100% of its own positioning so it can't silently break.
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
          const camDir = camera.position.clone().normalize();

          labels.forEach((label, i) => {
            const el = elRefs.current[i];
            if (!el) return;

            const c = g.getCoords(label.lat, label.lng, LABEL_ALTITUDE);
            const vec = new THREE.Vector3(c.x, c.y, c.z);

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
