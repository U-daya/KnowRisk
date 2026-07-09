export interface CountryInfo {
  longitude: number;
  latitude: number;
  flag: string;
}

// Exactly the nine countries present in data/suppliers.json
export const COUNTRY_COORDS: Record<string, CountryInfo> = {
  "Taiwan":      { longitude: 120.9605, latitude: 23.6978,  flag: "🇹🇼" },
  "South Korea": { longitude: 127.7669, latitude: 35.9078,  flag: "🇰🇷" },
  "Japan":       { longitude: 138.2529, latitude: 36.2048,  flag: "🇯🇵" },
  "Netherlands": { longitude: 5.2913,   latitude: 52.1326,  flag: "🇳🇱" },
  "USA":         { longitude: -95.7129, latitude: 37.0902,  flag: "🇺🇸" },
  "China":       { longitude: 104.1954, latitude: 35.8617,  flag: "🇨🇳" },
  "Germany":     { longitude: 10.4515,  latitude: 51.1657,  flag: "🇩🇪" },
  "Malaysia":    { longitude: 101.9758, latitude: 4.2105,   flag: "🇲🇾" },
  "Israel":      { longitude: 34.8516,  latitude: 31.0461,  flag: "🇮🇱" },
};
