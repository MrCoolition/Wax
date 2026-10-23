"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";
import {
  ArrowUpRight,
  ArrowUp,
  ArrowDown,
  Check,
  Clock3,
  Compass,
  Disc3,
  ExternalLink,
  LocateFixed,
  MapPin,
  Navigation,
  Plus,
  Route,
  RefreshCw,
  Search,
  Store,
  X,
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
import type { SavedShop, ShopSnapshot } from "@/lib/shop-notebook";
import {
  SavedShops,
  ShopNotebookEditor,
  ShopPeople,
  ShopRating,
  useShopNotebook,
} from "./shop-notebook";
import {
  MAX_DIG_STOPS,
  digRouteUrl,
  moveDigStop,
  newShopsInExpandedArea,
} from "@/lib/shop-pathfinder";
import styles from "./shop-pathfinder.module.css";

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
  100: "60 miles",
};

function mapsSearch(area: ShopArea | null, label?: string) {
  const query = area
    ? `record shops near ${area.latitude},${area.longitude}`
    : `record shops near ${label || "me"}`;
  return `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(query)}`;
}

function ShopCard({
  shop,
  index,
  saved,
  notebookReady,
  onEdit,
  routePosition,
  routeFull,
  onRouteToggle,
}: {
  shop: RecordShop;
  index: number;
  saved?: SavedShop;
  notebookReady: boolean;
  onEdit: (shop: ShopSnapshot, saved: SavedShop | null) => void;
  routePosition: number;
  routeFull: boolean;
  onRouteToggle: (shop: RecordShop) => void;
}) {
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
      {saved && (
        <div className="shop-card-personal">
          <ShopRating rating={saved.rating} />
          <ShopPeople employees={saved.employees} />
        </div>
      )}
      <div className="shop-card-actions">
        <a
          className="button button-primary"
          href={directions}
          target="_blank"
          rel="noopener noreferrer"
        >
          Directions <ArrowUpRight size={16} />
        </a>
        <Button
          className={styles.addStop}
          variant="secondary"
          disabled={routeFull && routePosition < 0}
          aria-pressed={routePosition >= 0}
          aria-label={
            routePosition >= 0 ? "Remove from dig route" : "Add to dig route"
          }
          onClick={() => onRouteToggle(shop)}
        >
          {routePosition >= 0 ? <Check size={14} /> : <Plus size={14} />}
          {routePosition >= 0
            ? `Stop ${routePosition + 1} · Added`
            : routeFull
              ? "Route full"
              : "Add to dig route"}
        </Button>
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
      </div>
      <Button
        className="shop-notebook-trigger"
        variant="quiet"
        disabled={!notebookReady}
        onClick={() =>
          onEdit(
            {
              id: shop.id,
              name: shop.name,
              latitude: shop.latitude,
              longitude: shop.longitude,
              address: shop.address,
              website: shop.website,
            },
            saved ?? null,
          )
        }
      >
        {saved ? "Your shop notes" : "Rate & add people"}
        <ArrowUpRight size={14} />
      </Button>
    </li>
  );
}

function DigRoute({
  center,
  label,
  stops,
  onRemove,
  onMove,
  onClear,
}: {
  center: ShopArea;
  label: string;
  stops: RecordShop[];
  onRemove: (shop: RecordShop) => void;
  onMove: (index: number, direction: -1 | 1) => void;
  onClear: () => void;
}) {
  const href = digRouteUrl(center, stops);
  return (
    <section className={styles.digRoute} aria-label="Your dig route">
      <div className={styles.routeHeading}>
        <div className={styles.routeTitle}>
          <span className={styles.routeIcon}>
            <Route size={22} />
          </span>
          <div>
            <span className="eyebrow">MAKE A DAY OF IT</span>
            <h3>
              Your dig route<span>.</span>
            </h3>
          </div>
        </div>
        <span className={styles.routeCount}>
          {stops.length}
          <span> / {MAX_DIG_STOPS} STOPS</span>
        </span>
      </div>
      {stops.length ? (
        <>
          <div className={styles.routeStart}>
            <MapPin size={12} /> Starting near {label}
          </div>
          <ol className={styles.routeStops}>
            {stops.map((stop, index) => (
              <li key={stop.id}>
                <span className={styles.stopNumber}>{index + 1}</span>
                <div className={styles.stopDetails}>
                  <strong>{stop.name}</strong>
                  <span>
                    {stop.address ||
                      "Use directions for the shop’s mapped location."}
                  </span>
                </div>
                <div className={styles.stopActions}>
                  <button
                    type="button"
                    aria-label={`Move ${stop.name} earlier`}
                    disabled={index === 0}
                    onClick={() => onMove(index, -1)}
                  >
                    <ArrowUp size={14} />
                  </button>
                  <button
                    type="button"
                    aria-label={`Move ${stop.name} later`}
                    disabled={index === stops.length - 1}
                    onClick={() => onMove(index, 1)}
                  >
                    <ArrowDown size={14} />
                  </button>
                  <button
                    type="button"
                    aria-label={`Remove ${stop.name} from dig route`}
                    onClick={() => onRemove(stop)}
                  >
                    <X size={15} />
                  </button>
                </div>
              </li>
            ))}
          </ol>
          <div className={styles.routeFooter}>
            <button
              type="button"
              className={styles.clearRoute}
              onClick={onClear}
            >
              Clear route
            </button>
            {href && (
              <a
                className={styles.openRoute}
                href={href}
                target="_blank"
                rel="noopener noreferrer"
              >
                Open dig route <ArrowUpRight size={16} />
              </a>
            )}
          </div>
          <p className={styles.routePrivacy}>
            Google Maps opens with your approximate starting area and chosen
            stop order. It calculates the drive; you can adjust the start there.
          </p>
        </>
      ) : (
        <div className={styles.routeEmpty}>
          <div aria-hidden="true">
            {[1, 2, 3, 4].map((number) => (
              <span key={number}>{number}</span>
            ))}
          </div>
          <p>
            Add up to four shops below. Put them in your order, then take the
            whole dig to Google Maps.
          </p>
        </div>
      )}
      <p className={styles.routeSession}>
        This route stays in this tab until you leave or choose a new area.
      </p>
    </section>
  );
}

export function ShopsView() {
  const notebook = useShopNotebook();
  const [editingShop, setEditingShop] = useState<{
    shop: ShopSnapshot;
    saved: SavedShop | null;
  } | null>(null);
  const [query, setQuery] = useState("");
  const [radius, setRadius] = useState<ShopRadiusKm>(10);
  const [places, setPlaces] = useState<ShopPlace[]>([]);
  const [placeQuery, setPlaceQuery] = useState("");
  const [selected, setSelected] = useState<SelectedArea | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [busy, setBusy] = useState<"" | "location" | "places" | "nearby">("");
  const [error, setError] = useState("");
  const [retry, setRetry] = useState<Retry | null>(null);
  const [shopFilter, setShopFilter] = useState("");
  const [digStops, setDigStops] = useState<RecordShop[]>([]);
  const [routeMessage, setRouteMessage] = useState("");
  const [expansionMessage, setExpansionMessage] = useState("");
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
    setExpansionMessage("");
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
      const added = newShopsInExpandedArea(result, response);
      if (added === 0) {
        setExpansionMessage(
          `${response.shops.length === result?.shops.length ? `Same ${response.shops.length} shops in the wider radius. ` : ""}No additional verified vinyl shops were found in the available listings within ${radiusLabels[nextRadius]}.`,
        );
      } else if (added !== null) {
        setExpansionMessage(
          `${added} more ${added === 1 ? "shop" : "shops"} to dig through in this wider search.`,
        );
      }
      if (
        result &&
        (response.center.latitude !== result.center.latitude ||
          response.center.longitude !== result.center.longitude)
      ) {
        setDigStops([]);
        setShopFilter("");
        setRouteMessage("New area selected. Your dig route has been cleared.");
      }
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
  const filterText = shopFilter.trim().toLocaleLowerCase();
  const filteredShops = filterText
    ? shops.filter((shop) =>
        `${shop.name} ${shop.address || ""}`
          .toLocaleLowerCase()
          .includes(filterText),
      )
    : shops;
  const savedById = new Map(
    notebook.shops.map((entry) => [entry.shop.id, entry]),
  );
  function editShop(shop: ShopSnapshot, saved: SavedShop | null) {
    if (notebook.status === "ready") setEditingShop({ shop, saved });
  }
  function toggleRouteShop(shop: RecordShop) {
    const present = digStops.some((stop) => stop.id === shop.id);
    if (!present && digStops.length >= MAX_DIG_STOPS) return;
    setDigStops((current) =>
      present
        ? current.filter((stop) => stop.id !== shop.id)
        : [...current, shop],
    );
    setRouteMessage(
      present
        ? `${shop.name} removed from your dig route.`
        : `${shop.name} added as stop ${digStops.length + 1}.`,
    );
  }
  return (
    <div className="shops-view">
      <header className="shops-hero">
        <div className="shops-hero-copy">
          <span className="eyebrow">
            <span /> WAX PATHFINDER / OUT IN THE REAL WORLD
          </span>
          <h1>
            Go find
            <br />
            the good stuff<span>.</span>
          </h1>
          <p>
            Some records find you. The rest are waiting in a crate around the
            corner. Find your spots. Pick your stops. Make a day of it.
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
          <fieldset className={`shops-radius ${styles.radiusPicker}`}>
            <legend>How far?</legend>
            <div>
              {SHOP_RADII_KM.map((value) => (
                <label key={value}>
                  <input
                    type="radio"
                    name="shops-radius"
                    aria-label={radiusLabels[value]}
                    checked={radius === value}
                    value={value}
                    onChange={() => {
                      setRadius(value);
                      currentRadius.current = value;
                      if (selected) void findNearby(selected, value);
                    }}
                  />
                  <span>
                    {radiusLabels[value]}
                    {value === 100 && <small>DAY TRIP</small>}
                  </span>
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

      <SavedShops notebook={notebook} onEdit={editShop} />
      <span className="sr-only" role="status">
        {routeMessage}
      </span>

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
          {expansionMessage && (
            <p className={styles.expansionNotice} role="status">
              <Compass size={18} />
              <span>{expansionMessage}</span>
            </p>
          )}
          {result.discovery?.mapAvailable === false && (
            <div className={styles.partialNotice} role="status">
              <div>
                <strong>Map search is taking a break.</strong>
                <p>
                  These results come from our verified record retailer
                  directory. Some local shops may be missing until map listings
                  are available again.
                </p>
              </div>
              <Button
                variant="secondary"
                busy={busy === "nearby"}
                onClick={() =>
                  void findNearby(
                    { ...result.center, label: result.label },
                    result.radiusKm,
                  )
                }
              >
                <RefreshCw size={14} /> Retry map search
              </Button>
            </div>
          )}
          <DigRoute
            center={result.center}
            label={result.label}
            stops={digStops}
            onRemove={toggleRouteShop}
            onMove={(index, direction) => {
              setDigStops((current) => moveDigStop(current, index, direction));
              setRouteMessage(
                `${digStops[index].name} moved ${direction === -1 ? "earlier" : "later"} in your route.`,
              );
            }}
            onClear={() => {
              setDigStops([]);
              setRouteMessage("Your dig route is clear.");
            }}
          />
          {shops.length ? (
            <>
              <div className={styles.resultsToolbar}>
                <label htmlFor="shops-filter">
                  <span>Filter these shops</span>
                  <div>
                    <Search size={16} />
                    <input
                      id="shops-filter"
                      type="search"
                      value={shopFilter}
                      onChange={(event) => setShopFilter(event.target.value)}
                      placeholder="Shop name, neighborhood, street…"
                      maxLength={120}
                      autoComplete="off"
                    />
                  </div>
                </label>
                <p role="status">
                  Showing <strong>{filteredShops.length}</strong> of{" "}
                  {shops.length} shops
                </p>
              </div>
              {!filteredShops.length && (
                <div className={styles.noFilterMatches}>
                  <Search size={23} />
                  <p>No shops match “{shopFilter}” in these results.</p>
                  <Button variant="secondary" onClick={() => setShopFilter("")}>
                    Clear shop filter
                  </Button>
                </div>
              )}
              <ol className="shops-grid">
                {filteredShops.map((shop) => (
                  <ShopCard
                    key={shop.id}
                    shop={shop}
                    index={shops.findIndex((item) => item.id === shop.id)}
                    saved={savedById.get(shop.id)}
                    notebookReady={notebook.status === "ready"}
                    onEdit={editShop}
                    routePosition={digStops.findIndex(
                      (stop) => stop.id === shop.id,
                    )}
                    routeFull={digStops.length >= MAX_DIG_STOPS}
                    onRouteToggle={toggleRouteShop}
                  />
                ))}
              </ol>
              <p className="shops-results-note">
                Listings need evidence of vinyl or record sales. Verified
                retailer listings supplement the map search. Unconfirmed music
                stores are left out. Distances are approximate and measured in a
                straight line, not by road. Check listed hours with the shop
                before heading out.
                {result.limited &&
                  " Results are limited; try a smaller radius for a more focused search."}
              </p>
            </>
          ) : (
            <div className="shops-empty">
              <Compass size={42} strokeWidth={1.3} />
              <h3>No matching record shops in this area yet.</h3>
              <p>
                Listings need evidence of vinyl or record sales. Community data
                isn’t exhaustive. Try a wider radius, search another area, or
                keep digging on Google Maps.
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
          . We also include verified record retailer listings. Results can be
          incomplete or out of date.
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
      {editingShop && (
        <ShopNotebookEditor
          shop={editingShop.shop}
          saved={editingShop.saved}
          notebook={notebook}
          onClose={() => setEditingShop(null)}
        />
      )}
    </div>
  );
}
