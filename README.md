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

Open **Record shops** at `/shops`, then choose **Use my current location** or search for a city or postal code. Location is requested only after clicking the button. Manual searches return area choices, including country/state information for ambiguous names and postal codes. Searches cover 10, 25, or 50 kilometres and show mapped recorded-music shops, approximate straight-line distances, address details, source listings, and Google Maps directions. Instrument-only stores are excluded by category; community data can still be incomplete or outdated, and vinyl stock is not guaranteed.

The browser rounds device coordinates to two decimal places before sending them in an authenticated POST body. The server applies the same rounding and sends only the approximate area to Photon; user identity, browser IP, and precise device coordinates are not forwarded. WAX stores shared, expiring place/shop results without an owner ID or location history. Place results are cached for seven days and shop results for six hours. Expired entries are removed on subsequent searches. Per-collector limits, a shared provider throttle, bounded response sizes, and timeouts protect the service. Provider errors preserve the selected area and offer retry or an external Maps search.

[Photon](https://github.com/komoot/photon#demo-server) supplies OpenStreetMap data without an API key for reasonable project usage. The page attributes [OpenStreetMap contributors](https://www.openstreetmap.org/copyright). `PHOTON_API_URL` and `PHOTON_REVERSE_API_URL` can select alternative HTTPS Photon-compatible endpoints; the defaults are `https://photon.komoot.io/api/` and `https://photon.komoot.io/reverse`. As usage grows, move to a dedicated provider or self-hosted Photon instance. No public map tiles, paid Places API, or AI inference is required.

`node --env-file-if-exists=.env.local scripts/smoke-shops.mjs` verifies the live application API, Neon cache, and provider results against a local app backed by the isolated development database. For a protected preview, set `WAX_TEST_URL` to its exact generated URL, use the matching `wax_preview` database credentials, and supply the linked project's short-lived development `VERCEL_OIDC_TOKEN`. It creates and cleans up its own temporary invited collector; it cannot run against the production database.

## Migrations and deployment

Checked-in SQL lives in `migrations/`. `npm run db:migrate` takes a transaction-scoped Postgres advisory lock, applies pending migrations together in one transaction, and records their checksums in `wax.schema_migrations`. This lock works with Neon's pooled connections. Add a new migration to change an existing database; do not edit an applied migration.

Vercel should use Node.js 24, `npm ci`, and the build command `npm run vercel:build`. This verifies the environment's database, applies its migrations, and builds Next.js. Assign production, preview, and development database URLs separately in Vercel. Keep preview deployment protection enabled; production uses WAX authentication so invited collectors do not need Vercel accounts.

GitHub Actions runs installation, type checks, tests, and a production build without application secrets. It does not run migrations or deploy. The connected Vercel project handles Git deployments. A failed application build does not reverse already applied migrations, so schema changes must remain compatible with the preceding deployment.

## Privacy and data behavior

- Every collection operation checks the authenticated collector. Sharing grants read-only access to a selected crate; purchase prices, private notes, original uploads, exports, and Wizard context remain private. Owners can revoke a grant.
- Releases and owned copies are separate. Intentional duplicates remain separate copies; save idempotency prevents accidental duplicates from repeated requests. Updates use versions to detect conflicting edits.
- Provider lookup payloads live in the expiring catalog cache. Release rows retain identity; per-copy fields retain the collector's reviewed text and corrections. Artwork URLs are not retained as copy fields, and catalog refresh never replaces saved text. These are implemented data-retention boundaries, not a general assertion of compliance with every provider term.
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
