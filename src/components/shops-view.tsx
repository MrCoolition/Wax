"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";
import {
  ArrowUpRight,
  Clock3,
  Compass,
  Disc3,
  ExternalLink,
  LocateFixed,
  MapPin,
  Navigation,
  RefreshCw,
  Search,
  Store,
} from "lucide-react";
import {
  approximateShopArea,
  SHOP_RADII_KM,
  type NearbyShops,
  type RecordShop,
  type ShopArea,
  type ShopPlace,
  type ShopRadiusKm,
} from "@/lib/shops";
import { api } from "./library-provider";
import { Button, ErrorNotice } from "./ui";

type SelectedArea = ShopArea & { label: string };
type Result = NearbyShops & { label: string };
type Retry =
  | { kind: "location" }
  | { kind: "places"; query: string }
  | { kind: "nearby"; area: SelectedArea; radius: ShopRadiusKm };
const radiusLabels: Record<ShopRadiusKm, string> = {
  10: "6 miles",
  25: "15 miles",
  50: "30 miles",
};

function mapsSearch(area: ShopArea | null, label?: string) {
  const query = area
    ? `record shops near ${area.latitude},${area.longitude}`
    : `record shops near ${label || "me"}`;
  return `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(query)}`;
}

function ShopCard({ shop, index }: { shop: RecordShop; index: number }) {
  const miles = shop.distanceKm * 0.621371;
  const distance =
    miles < 0.1 ? "Under 0.1 mi" : `About ${miles.toFixed(1)} mi`;
  const directions = `https://www.google.com/maps/dir/?api=1&destination=${encodeURIComponent(`${shop.latitude},${shop.longitude}`)}`;
  return (
    <li className="shop-card">
      <div className="shop-card-heading">
        <span className="shop-number" aria-hidden="true">
          {String(index + 1).padStart(2, "0")}
        </span>
        <span className="shop-distance">
          <Navigation size={12} /> {distance}
        </span>
      </div>
      <div className="shop-card-name">
        <span className="eyebrow">ANOTHER CRATE TO DIG</span>
        <h3>{shop.name}</h3>
      </div>
      <div className="shop-card-details">
        <p>
          <MapPin size={15} />
          <span>
            {shop.address ||
              "Address details unavailable. Use directions to find the mapped location."}
          </span>
        </p>
        {shop.openingHours && (
          <p>
            <Clock3 size={15} />
            <span>
              <strong>Listed hours</strong>
              {shop.openingHours}
            </span>
          </p>
        )}
        {shop.phone && <p className="shop-phone">Phone: {shop.phone}</p>}
      </div>
      <div className="shop-card-actions">
        <a
          className="button button-primary"
          href={directions}
          target="_blank"
          rel="noopener noreferrer"
        >
          Directions <ArrowUpRight size={16} />
        </a>
        {shop.website && (
          <a
            className="shop-text-link"
            href={shop.website}
            target="_blank"
            rel="noopener noreferrer"
          >
            Website <ExternalLink size={13} />
          </a>
        )}
        <a
          className="shop-source"
          href={shop.osmUrl}
          target="_blank"
          rel="noopener noreferrer"
          aria-label={`View ${shop.name} on OpenStreetMap`}
        >
          OSM listing <ArrowUpRight size={12} />
        </a>
      </div>
    </li>
  );
}

export function ShopsView() {
  const [query, setQuery] = useState("");
  const [radius, setRadius] = useState<ShopRadiusKm>(10);
  const [places, setPlaces] = useState<ShopPlace[]>([]);
  const [placeQuery, setPlaceQuery] = useState("");
  const [selected, setSelected] = useState<SelectedArea | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [busy, setBusy] = useState<"" | "location" | "places" | "nearby">("");
  const [error, setError] = useState("");
  const [retry, setRetry] = useState<Retry | null>(null);
  const sequence = useRef(0);
  const activeRequest = useRef<AbortController | null>(null);
  const currentRadius = useRef(radius);

  useEffect(
    () => () => {
      sequence.current++;
      activeRequest.current?.abort();
    },
    [],
  );

  function beginRequest() {
    activeRequest.current?.abort();
    const controller = new AbortController();
    activeRequest.current = controller;
    const id = ++sequence.current;
    setError("");
    setRetry(null);
    return { id, signal: controller.signal };
  }

  async function findNearby(
    area: SelectedArea,
    nextRadius = currentRadius.current,
  ) {
    const safeArea = { ...approximateShopArea(area), label: area.label };
    const { id, signal } = beginRequest();
    setBusy("nearby");
    setSelected(safeArea);
    setPlaces([]);
    setPlaceQuery("");
    try {
      const response = await api<NearbyShops>("/api/shops/nearby", {
        method: "POST",
        signal,
        body: JSON.stringify({
          latitude: safeArea.latitude,
          longitude: safeArea.longitude,
          radiusKm: nextRadius,
        }),
      });
      if (id !== sequence.current) return;
      setResult({ ...response, label: safeArea.label });
    } catch (err) {
      if (id !== sequence.current) return;
      setError(
        err instanceof Error
          ? err.message
          : "We couldn’t look up shops just now. Please try again.",
      );
      setRetry({ kind: "nearby", area: safeArea, radius: nextRadius });
    } finally {
      if (id === sequence.current) setBusy("");
    }
  }

  async function findPlaces(searchQuery: string) {
    const trimmed = searchQuery.trim();
    if (!trimmed) return;
    const { id, signal } = beginRequest();
    setBusy("places");
    setPlaces([]);
    setPlaceQuery("");
    try {
      const response = await api<{ places: ShopPlace[] }>("/api/shops/places", {
        method: "POST",
        signal,
        body: JSON.stringify({ query: trimmed }),
      });
      if (id !== sequence.current) return;
      setPlaceQuery(trimmed);
      setPlaces(response.places);
    } catch (err) {
      if (id !== sequence.current) return;
      setError(
        err instanceof Error
          ? err.message
          : "We couldn’t find that area. Please try again.",
      );
      setRetry({ kind: "places", query: trimmed });
    } finally {
      if (id === sequence.current) setBusy("");
    }
  }

  function locate() {
    const { id } = beginRequest();
    setBusy("location");
    setPlaces([]);
    setPlaceQuery("");
    if (!navigator.geolocation) {
      setError(
        "This browser can’t share your location. Enter a city or postal code below to find your next dig.",
      );
      setBusy("");
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (position) => {
        if (id !== sequence.current) return;
        // Do not store, log, or send the precise device position.
        const area = approximateShopArea({
          latitude: position.coords.latitude,
          longitude: position.coords.longitude,
        });
        void findNearby({ ...area, label: "Your approximate location" });
      },
      (locationError) => {
        if (id !== sequence.current) return;
        const message =
          locationError.code === 1
            ? "Location access is off. You can allow it in your browser settings, or enter a city or postal code below."
            : locationError.code === 3
              ? "Finding your location took too long. Try again, or enter a city or postal code below."
              : "Your location isn’t available right now. Try again, or enter a city or postal code below.";
        setError(message);
        setRetry({ kind: "location" });
        setBusy("");
      },
      { enableHighAccuracy: false, timeout: 10_000, maximumAge: 300_000 },
    );
  }

  function submitPlace(event: FormEvent) {
    event.preventDefault();
    void findPlaces(query);
  }

  function retryLast() {
    if (retry?.kind === "location") locate();
    else if (retry?.kind === "places") void findPlaces(retry.query);
    else if (retry?.kind === "nearby")
      void findNearby(retry.area, retry.radius);
  }

  const shops =
    result?.shops.slice().sort((a, b) => a.distanceKm - b.distanceKm) || [];
  return (
    <div className="shops-view">
      <header className="shops-hero">
        <div className="shops-hero-copy">
          <span className="eyebrow">
            <span /> OUT IN THE REAL WORLD
          </span>
          <h1>
            Go find
            <br />
            the good stuff<span>.</span>
          </h1>
          <p>
            Some records find you. The rest are waiting in a crate around the
            corner.
          </p>
        </div>
        <div className="shops-hero-art" aria-hidden="true">
          <span className="shops-art-caption">FOLLOW YOUR EARS</span>
          <svg className="shops-route" viewBox="0 0 260 190" fill="none">
            <path
              d="M32 150H110V80H220V25"
              stroke="currentColor"
              strokeWidth="3"
              strokeDasharray="5 7"
            />
            <circle cx="32" cy="150" r="8" fill="currentColor" />
            <circle
              cx="220"
              cy="25"
              r="12"
              stroke="currentColor"
              strokeWidth="3"
            />
          </svg>
          <div className="shops-art-record">
            <Disc3 strokeWidth={0.8} />
            <span>
              THE NEXT
              <br />
              GREAT FIND
            </span>
          </div>
          <span className="shops-art-footer">EST. WHENEVER YOU GET THERE.</span>
        </div>
      </header>

      <section className="shops-finder" aria-labelledby="shops-finder-heading">
        <div className="shops-finder-top">
          <div>
            <span className="eyebrow">PICK YOUR STARTING POINT</span>
            <h2 id="shops-finder-heading">Where are we digging?</h2>
          </div>
          <Button onClick={locate} busy={busy === "location"}>
            <LocateFixed size={17} />
            {busy === "location"
              ? "Finding your area…"
              : "Use my current location"}
          </Button>
        </div>
        <div className="shops-finder-fields">
          <form className="shops-place-form" onSubmit={submitPlace}>
            <label htmlFor="shops-place">Or enter a city or postal code</label>
            <div>
              <input
                id="shops-place"
                name="place"
                placeholder="e.g. Brooklyn, NY"
                autoComplete="off"
                maxLength={120}
                required
                value={query}
                onChange={(event) => {
                  setQuery(event.target.value);
                  sequence.current++;
                  activeRequest.current?.abort();
                  setBusy("");
                  setPlaces([]);
                  setPlaceQuery("");
                  setError("");
                  setRetry(null);
                }}
              />
              <Button
                type="submit"
                variant="secondary"
                busy={busy === "places"}
                disabled={!query.trim()}
              >
                <Search size={16} />
                <span>Find area</span>
              </Button>
            </div>
          </form>
          <fieldset className="shops-radius">
            <legend>How far?</legend>
            <div>
              {SHOP_RADII_KM.map((value) => (
                <label key={value}>
                  <input
                    type="radio"
                    name="shops-radius"
                    checked={radius === value}
                    value={value}
                    onChange={() => {
                      setRadius(value);
                      currentRadius.current = value;
                      if (selected) void findNearby(selected, value);
                    }}
                  />
                  <span>{radiusLabels[value]}</span>
                </label>
              ))}
            </div>
          </fieldset>
        </div>
        <p className="shops-privacy">
          Your precise location stays in your browser. We share only your
          approximate area to find shops; we don’t save your location to your
          collection.
        </p>
        {error && (
          <div className="shops-error">
            <ErrorNotice>{error}</ErrorNotice>
            {retry && (
              <Button variant="secondary" onClick={retryLast}>
                <RefreshCw size={14} />
                Try again
              </Button>
            )}
          </div>
        )}
        {placeQuery && (
          <div className="shops-places" aria-live="polite">
            {places.length ? (
              <>
                <h3>Choose your area</h3>
                <ul>
                  {places.map((place) => (
                    <li key={place.id}>
                      <button
                        type="button"
                        onClick={() =>
                          void findNearby({
                            ...place,
                            ...approximateShopArea(place),
                          })
                        }
                      >
                        <MapPin size={17} />
                        <span>{place.label}</span>
                        <ArrowUpRight size={17} />
                      </button>
                    </li>
                  ))}
                </ul>
              </>
            ) : (
              <p>
                No area found for “{placeQuery}”. Try adding your region or
                country, or use your current location.
              </p>
            )}
          </div>
        )}
      </section>

      <div className="shops-status" role="status" aria-live="polite">
        {busy === "places" ? (
          <>
            <Search size={17} /> Finding matching areas…
          </>
        ) : busy === "nearby" ? (
          <>
            <Disc3 size={18} className="spin" /> Digging up record shops…
          </>
        ) : !busy && result ? (
          <span className="sr-only">
            {shops.length} record shops found near {result.label}.
          </span>
        ) : null}
      </div>
      {result ? (
        <section
          className="shops-results"
          aria-labelledby="shops-results-heading"
          aria-busy={busy === "nearby"}
        >
          <div className="shops-results-heading">
            <div>
              <span className="eyebrow">
                NEAREST FIRST · {radiusLabels[result.radiusKm].toUpperCase()}{" "}
                RADIUS, APPROX.
              </span>
              <h2 id="shops-results-heading">
                {shops.length
                  ? `${shops.length}${result.limited ? "+" : ""} places to dig`
                  : "A little further off the beaten track."}
              </h2>
              <p>
                <MapPin size={14} />
                {result.label}
              </p>
            </div>
            <a
              className="shop-text-link"
              href={mapsSearch(result.center)}
              target="_blank"
              rel="noopener noreferrer"
            >
              Search Google Maps <ArrowUpRight size={15} />
            </a>
          </div>
          {shops.length ? (
            <>
              <ol className="shops-grid">
                {shops.map((shop, index) => (
                  <ShopCard key={shop.id} shop={shop} index={index} />
                ))}
              </ol>
              <p className="shops-results-note">
                Distances are approximate and measured in a straight line, not
                by road. Check listed hours with the shop before heading out.
                {result.limited &&
                  " Results are limited; try a smaller radius for a more focused search."}
              </p>
            </>
          ) : (
            <div className="shops-empty">
              <Compass size={42} strokeWidth={1.3} />
              <h3>No mapped record shops in this area yet.</h3>
              <p>
                Community listings aren’t exhaustive. Try a wider radius, search
                another area, or keep digging on Google Maps.
              </p>
              <a
                className="button button-secondary"
                href={mapsSearch(result.center)}
                target="_blank"
                rel="noopener noreferrer"
              >
                Look on Google Maps <ArrowUpRight size={16} />
              </a>
            </div>
          )}
        </section>
      ) : (
        !busy &&
        !placeQuery && (
          <section className="shops-start">
            <Store size={32} strokeWidth={1.3} />
            <div>
              <h2>Your next favorite shop is out there.</h2>
              <p>
                Choose an area to find record shops, get directions, and make a
                day of it.
              </p>
            </div>
            <span className="shops-start-stamp" aria-hidden="true">
              SUPPORT
              <br />
              YOUR LOCAL
            </span>
          </section>
        )
      )}
      {error && (
        <a
          className="shop-text-link shops-fallback"
          href={mapsSearch(
            retry?.kind === "places" || retry?.kind === "location"
              ? null
              : selected,
            retry?.kind === "location" ? "me" : query,
          )}
          target="_blank"
          rel="noopener noreferrer"
        >
          Keep digging on Google Maps <ArrowUpRight size={15} />
        </a>
      )}
      <footer className="shops-attribution">
        <p>
          Shop listings ©{" "}
          <a
            href="https://www.openstreetmap.org/copyright"
            target="_blank"
            rel="noopener noreferrer"
          >
            OpenStreetMap contributors
          </a>
          . Community data can be incomplete or out of date.
        </p>
        <p>
          Your approximate search area is shared with{" "}
          <a
            href="https://photon.komoot.io"
            target="_blank"
            rel="noopener noreferrer"
          >
            Photon <ArrowUpRight size={11} />
          </a>
          , our place-search provider. Check with the shop before making the
          trip.
        </p>
      </footer>
    </div>
  );
}
