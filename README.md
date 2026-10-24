# WAX

Serious collection tools. Record-shop soul.

WAX is a private vinyl library for invited collectors: artwork and table browsing, physical copies, crates, shelf locations, search/barcode/photo capture, listening history, CSV export, printable record QR labels, selective crate sharing, nearby record shops, and the Wax Wizard. Real collections start empty; fixtures belong to tests.

The app uses Next.js App Router, React, TypeScript, Tailwind/Radix, Node.js 24, Neon Postgres, Drizzle, and Vercel. The former Streamlit application remains recoverable in Git history.

## Connections

- **Repository/deployment:** `MrCoolition/Wax` → existing Vercel project `coolition/wax`, configured as Next.js.
- **Database:** the existing production Neon database is preserved. New tables live in the `wax` schema; migrations do not delete legacy collection data. Development and preview use separate `wax_development` and `wax_preview` databases with matching restricted roles in the same Neon project. These are separate databases, not Neon branches; preview deployments share the preview database.
- **Authentication:** Better Auth with email/password and owner-issued invitations. This implementation does not require Clerk.
- **Catalog:** server-side Discogs release search and exact-release lookup, with optional authenticated access. Catalog results are attributed to Discogs and cached for up to six hours. Saved collection text is the collector's reviewed snapshot; catalog refresh does not overwrite it. Discogs artwork is displayed only from release metadata fetched within the last six hours; expired artwork becomes a placeholder until refreshed.
- **AI:** Vercel AI SDK with the direct OpenAI provider and the existing OpenAI key. The default is `gpt-5.4-mini`. AI Gateway is not required by this implementation.
- **Photos:** private Vercel Blob storage; authenticated application routes serve uploaded images.

## Finding your next record

**Add some wax** starts with album discovery: type an artist and title together, or use the separate fields. Album results open paginated vinyl pressings with country, label, and pressing-year filters. “Beatles White Album” resolves to its catalog title, **The Beatles**, before choosing an edition. Catalog numbers and barcodes still search releases directly. Discogs may match a secondary label even when the result card lists only its primary label.

Hand-entered artist/title fields suggest album matches after a short pause. Selecting a match loads catalog details for review while retaining the physical copy's notes, condition, price, photos, and crates. Photo clues can be corrected through the same finder. Existing manual copies can use **Edit this copy → Find catalog matches**. Nothing is saved or pressing-confirmed automatically; the collector reviews and saves explicitly.

**Wax Wizard** answers buying and discovery questions beyond the collection, including for new collectors with empty libraries. It follows conversation corrections, uses bounded read-only Discogs album lookups, and returns verified album cards with reasons and links into pressing exploration. Only actual owned copies named in the answer or edit proposal appear as collection links. It has no live stock, marketplace prices, or web-search tool; catalog outages still allow general musical advice without invented catalog cards. Suggested collection edits continue to require explicit approval and ownership/version checks.

For deeper music and vinyl questions, five public catalog tools cover albums, vinyl pressings, release/track credits and identifiers, artist connections, and discographies. They share a maximum of six calls per answer. Retrieved source IDs are checked on the server before source links are displayed; the model cannot supply arbitrary URLs. Track-specific, album-level, and unresolved credits remain distinct. Company credits, matrix/runout text, and catalog notes are evidence about a listed edition, not proof of sound quality or a collector's exact pressing. General music knowledge supplies context and opinions; incomplete provider data stays incomplete.

The dedicated `scripts/smoke-catalog-search.mjs` checks real Discogs discovery and persisted manual enrichment in isolated development or preview. `scripts/smoke-wizard-discovery.mjs` additionally tests live AI buying advice, follow-up corrections, owned recommendations, reviewed edits, and musical credits; it requires `WAX_RUN_PAID_SMOKE=1` because it makes five billable model requests. Both create and clean up only disposable test collectors.

## Run locally

Use Node.js 24 and npm. Create `.env.local` from `.env.example`, set development credentials, then run:

```sh
npm ci
npm run db:verify
npm run db:migrate
npm run dev
```

Open `http://localhost:3000`. Do not point local development or preview deployments at the production database.

| Variable                   | Purpose                                                                                                                                                  |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DATABASE_URL`             | Postgres URL for the current environment. Only production supports existing `neon_connect` as a fallback. Preview and development require their own URL. |
| `BETTER_AUTH_SECRET`       | Independent random secret of at least 32 characters per environment.                                                                                     |
| `BETTER_AUTH_URL`          | Canonical application origin. Use `http://localhost:3000` locally; preview can fall back to `VERCEL_URL`.                                                |
| `OPENAI_API_KEY`           | OpenAI API key. Existing `chaz_gpt` is supported as a fallback.                                                                                          |
| `WAX_AI_MODEL`             | Optional OpenAI model override; defaults to `gpt-5.4-mini`.                                                                                              |
| `BLOB_READ_WRITE_TOKEN`    | Token for a private Blob store. A linked `BLOB_STORE_ID` with Vercel OIDC is also supported.                                                             |
| `DISCOGS_TOKEN`            | Optional Discogs personal-access token for authenticated lookup.                                                                                         |
| `WAX_OWNER_EMAIL`          | Email address bound to the initial owner invitation.                                                                                                     |
| `WAX_BOOTSTRAP_TOKEN`      | Random initial owner invitation token of at least 32 characters.                                                                                         |
| `WAX_OWNER_RECOVERY_TOKEN` | Optional random one-use recovery token for the existing owner; at least 32 characters.                                                                   |

Canonical environment names take precedence over their aliases. Runtime and migration checks require both the role and database to be `wax_preview` for `VERCEL_ENV=preview`, or `wax_development` for local/development processes. `VERCEL_ENV=production` selects production configuration. A missing or mismatched nonproduction URL fails before connection. Keep all values server-side and out of Git. `db:verify` checks database connectivity and reports whether AI and Blob configuration exists without printing credentials; it does not validate every provider's credentials.

## First owner and invitations

Set `WAX_OWNER_EMAIL` and a securely generated `WAX_BOOTSTRAP_TOKEN`, then run migrations for the intended environment. If no owner exists, the migration runner creates a one-use owner invitation valid for seven days.

If the unused owner invitation expires, rotate `WAX_BOOTSTRAP_TOKEN` to a new random value and run migrations again. Reusing the same token never extends its expiry; after an owner has registered, bootstrap configuration cannot replace that account.

Open `<app-origin>/join#token=<WAX_BOOTSTRAP_TOKEN>` privately, register using the configured email and a password of at least 12 characters, then sign in. The token is a URL fragment and only its hash is stored in Postgres. Treat the full link as a credential. After successful setup, remove the bootstrap token from deployment configuration.

The owner creates and revokes collector invitations in the application. Share those links with the intended recipients yourself; WAX does not send invitation emails. Public registration is disabled. Existing collectors receive their own private library.

## Owner password recovery

An invitation is consumed when the account is created. If the browser does not save the password, issuing another invitation cannot recover that account. An authorized operator can set a fresh cryptographically random `WAX_OWNER_RECOVERY_TOKEN` in the production environment, keep `WAX_OWNER_EMAIL` equal to the established owner's email, and run migrations through a production deployment. This creates a recovery link only for that existing owner, without changing any collection data.

Give the owner `<app-origin>/recover#token=<WAX_OWNER_RECOVERY_TOKEN>` privately, then remove the recovery token from Vercel configuration. The link expires after 24 hours and works once; only its SHA-256 digest is stored in Postgres. Reusing an old token cannot extend or reactivate it. A fresh token revokes older unused recovery links. Changing the password invalidates the account's existing sessions. The owner chooses and confirms their password on the recovery page and should save it in their browser or password manager; WAX cannot guarantee browser password saving.

## Nearby record shops

Open **Record shops** at `/shops`, then choose **Use my current location** or search for a city or postal code. Location is requested only after clicking the button. Manual searches return area choices, including country/state information for ambiguous names and postal codes. WAX Pathfinder searches 10, 25, 50, or 100 kilometres (approximately 6, 15, 30, or 60 miles), with approximate straight-line distances, addresses, and Google Maps directions. A wider search reports newly found shops or explains when available listings contain no additional matches. Filter by shop name/address and build an ordered route of up to four stops. The route lives only in the current page, survives radius/filter changes, and clears when a different area is selected; opening Google Maps sends the approximate starting area and selected stop coordinates. Google calculates the driving route.

Broad music-store categories alone do not qualify: current map metadata must explicitly indicate vinyl/records, a music-shop name must clearly identify a record/vinyl store, or checked sales evidence must match the exact business and location. Mixed retailers qualify with explicit vinyl evidence. Instrument, lesson, sheet-music, recording-studio, and control-vinyl signals are excluded without stronger sales evidence; closed locations and explicit no-vinyl metadata lose eligibility. Community data can still be incomplete or outdated, and current stock is not guaranteed. Expanding a radius cannot guarantee more results everywhere.

The browser rounds device coordinates to two decimal places before sending them in an authenticated POST body. The server applies the same rounding and sends only the approximate area to Photon; user identity, browser IP, and precise device coordinates are not forwarded. WAX stores shared, expiring place/shop results without an owner ID or location history. Place results are cached for seven days and shop results for six hours. Current public OpenStreetMap tags are fetched by candidate element IDs in at most three fixed-host batches and cached for seven days; contributor metadata is discarded. Current names, categories, closure/product tags, and node coordinates override older Photon details. Missing/deleted elements are excluded, and an unavailable metadata service yields a retryable error rather than unconfirmed results. Nearby cache keys include a filter version so previous broad music-store results cannot leak into the tightened results. Expired entries are removed on subsequent searches. Per-collector limits, a shared provider throttle, bounded response sizes, and timeouts protect the service. Provider errors preserve the selected area and offer retry or an external Maps search.

[Photon](https://github.com/komoot/photon#demo-server) supplies OpenStreetMap data without an API key for reasonable project usage. The page attributes [OpenStreetMap contributors](https://www.openstreetmap.org/copyright). `PHOTON_API_URL` and `PHOTON_REVERSE_API_URL` can select alternative HTTPS Photon-compatible endpoints; the defaults are `https://photon.komoot.io/api/` and `https://photon.komoot.io/reverse`. As usage grows, move to a dedicated provider or self-hosted Photon instance. No public map tiles, paid Places API, or AI inference is required.

The narrow checked-sales entries in `src/server/shop-vinyl.ts` retain their source URLs, check date (2026-10-22), OSM IDs, exact names, and coordinates: [RPM through Colemine Records' retailer directory](https://www.coleminerecords.com/blogs/news/indie-retail-online-store-directory) and [Magnolia Thunderpussy's vinyl catalog](https://thunderpussy.com/Genre/694768). They are not chain/name allowlists and cannot override a changed business identity, closed listing, or newer negative vinyl/format metadata. Search only requests known candidate IDs from the OSM API; it does not crawl merchant websites or use AI to guess stock.

`src/server/shop-directory.ts` supplements map gaps with six checked Central Ohio retailers. Each entry retains a sales/address source, review date, coordinate source, and coordinate accuracy. This is a limited editorial directory, not nationwide completeness. Census address-interpolated coordinates are approximate. Entries expire after 180 days and are merged outside the map cache, with exact-circle filtering and conservative duplicate matching. Recheck vinyl sales, current address, and any known OSM identity before renewing or adding an entry. Keep declared IDs stable so private ratings remain attached. During a map outage, eligible unmapped directory listings remain available with a partial-results notice and retry; failed provider results are not cached. Directory IDs are accepted by migration `0007_shop_directory.sql`, which preserves all existing notebook rows.

`node --env-file-if-exists=.env.local scripts/smoke-shops.mjs` verifies the live application API, Neon cache, and provider results against a local app backed by the isolated development database. For a protected preview, set `WAX_TEST_URL` to its exact generated URL, use the matching `wax_preview` database credentials, and supply the linked project's short-lived development `VERCEL_OIDC_TOKEN`. It creates and cleans up its own temporary invited collector; it cannot run against the production database.

### Your shop ratings and people

Each collector has a private shop notebook: save a 1–5 star rating, add employee names and roles, and choose from six illustrated avatars. Ratings can be cleared and people edited or removed. Saved shops remain available without repeating a nearby search. Changes are saved explicitly; failed requests retain the draft, and version checks prevent another tab from silently overwriting a newer edit. Removing a saved shop removes only that collector's rating and people.

`/api/shops/notebook` stores this information in `wax.shop_notebook`, scoped to the authenticated collector on every read and write. Shop ratings and employees are not public reviews, shared crate content, discovery-cache data, or Wizard context. Avatars are rendered locally without sending names to an image service. `scripts/smoke-shop-notebook.mjs` verifies persistence, updates, deletion, duplicate/stale-save protection, and isolation between two temporary collectors using the same development/preview safeguards as the shop-search smoke test.

## Song credits and the Artist Tree

Collection cards, shared crates, and year sorting use the album's original release year. The record detail and review forms keep **Album release year** and **Pressing year** separate; an unknown original year never falls back to a reissue date. Discogs-linked copies resolve the original year from their release's master entry, using the same six-hour provider cache. Collectors can correct either year independently. CSV exports and Wizard context label both fields explicitly.

On a collector's next page load, WAX enriches up to eight legacy copies missing the original-year field through an authenticated, owner-scoped POST. It updates only that field, checks copy versions, and preserves pressing information and personal data. Known missing metadata is recorded as unknown; provider failures remain retryable on a later load. Larger legacy collections continue in bounded batches on subsequent page loads. The notebook smoke test verifies this flow against live Discogs and Neon with disposable development/preview records.

Open a record and choose **Artist Tree** to pick a song, see its credited players, instruments, production, writing, and arrangements, then follow people into bands and other records. Credits without specific tracks remain separate from confirmed song credits; ambiguous track references are shown without guessing. Discogs may not list every contributor. Main artist billing alone never establishes an instrument credit.

The explorer follows one branch at a time, keeps a back trail, and pages through discographies. Master entries open their designated main release rather than treating master IDs as release IDs. Exact matches to the collector's own release IDs get an **In your collection** link. A record without a catalog link uses an explicit search-and-select step for exploration; this never updates the copy or confirms its pressing.

Four authenticated read-only routes serve the owned-record entry, public release credits, artist connections, and paginated discographies. Private copy ownership is checked before loading the entry. Only normalized public Discogs metadata enters the six-hour provider cache; collection badges remain private client state. Fixed provider paths, bounded responses, and rate limits protect the integration. No AI inference, new credentials, or schema migration is involved. `node --env-file-if-exists=.env.local scripts/smoke-explorer.mjs` verifies the live album → song → artist → related-release flow and two-collector isolation in development or a protected preview; it refuses production and cleans up its temporary collectors.

WAX's browser tab uses a vector record monogram with 16/32/48px ICO fallbacks. The phone shortcut uses a matching 180px record-label icon. Editable SVG sources live in `src/app/icon.svg` and `src/assets/wax-app-icon.svg`.

## Migrations and deployment

Checked-in SQL lives in `migrations/`. `npm run db:migrate` takes a transaction-scoped Postgres advisory lock, applies pending migrations together in one transaction, and records their checksums in `wax.schema_migrations`. This lock works with Neon's pooled connections. Add a new migration to change an existing database; do not edit an applied migration.

Vercel should use Node.js 24, `npm ci`, and the build command `npm run vercel:build`. This verifies the environment's database, applies its migrations, and builds Next.js. Assign production, preview, and development database URLs separately in Vercel. Keep preview deployment protection enabled; production uses WAX authentication so invited collectors do not need Vercel accounts.

GitHub Actions runs installation, type checks, tests, and a production build without application secrets. It does not run migrations or deploy. The connected Vercel project handles Git deployments. A failed application build does not reverse already applied migrations, so schema changes must remain compatible with the preceding deployment.

## Privacy and data behavior

- Every collection operation checks the authenticated collector. Sharing grants read-only access to a selected crate; purchase prices, private notes, original uploads, exports, and Wizard context remain private. Owners can revoke a grant.
- Releases and owned copies are separate. Intentional duplicates remain separate copies; save idempotency prevents accidental duplicates from repeated requests. Updates use versions to detect conflicting edits.
- Provider lookup payloads live in the expiring catalog cache. Release rows retain identity; per-copy fields retain the collector's reviewed text and corrections. Artwork URLs are not retained as copy fields, and catalog refresh never replaces saved text. Saved covers use an authenticated, stable `/api/catalog/{id}/artwork` resolver that renews expired provider metadata when the image is requested, including in shared crates and listening history. Transient image failures retry once, then recover on reconnect or return to the page. The notebook smoke test also verifies overnight cache expiry against a live Discogs cover. These are implemented data-retention boundaries, not a general assertion of compliance with every provider term.
- Photo recognition reads clues and looks up catalog candidates. Artwork never confirms a pressing automatically. Search failures and ambiguous matches support manual review.
- The Wizard receives the current collector's library context and, for recognition, the selected uploaded photo. These requests go to OpenAI with response storage disabled. Discogs receives catalog search terms. Neither provider receives database credentials.
- Wizard proposals do not change copies until approved. Approval rechecks ownership and versions and applies the whole proposal in a transaction; repeated approval does not apply it twice.
- Original photos require authentication even when a corresponding crate is shared. QR labels link to authenticated record pages.

## Checks and current limits

```sh
npm run typecheck
npm test
npm run build
```

Tests use fixtures and PGlite for SQL behavior, including ownership, duplicate handling, and reviewed Wizard edits. They do not call paid AI services. Browser verification is separate (`npm run test:browser`) and must use a dedicated nonproduction environment; successful unit tests do not establish real-device camera compatibility.

The first release targets collections under 1,000 records. Wizard context includes at most 1,000 copies and 500 characters of notes per copy; an edit proposal covers at most 20 copies. Uploads accept JPEG, PNG, and WebP up to 3 MiB; HEIC requires conversion. Camera scanning needs HTTPS or localhost and browser permission. Barcode and artwork matches may represent several pressings.

Invitations currently provide access without an email-delivery service. Email verification and self-service password recovery are not implemented; owner recovery requires an operator-issued link as described above. AI, Discogs, and Blob outages require retrying or manual entry. Uploaded photos are retained independently of copy deletion; a photo-retention/deletion interface is not yet available. Live provider access and physical iOS/Android camera behavior require deployment checks in addition to automated tests.

Listening-room tools, shop planning, commerce, public signup, marketplace valuation, and streaming playback are outside this release.
