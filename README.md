# WAX

Serious collection tools. Record-shop soul.

WAX is a private vinyl library for invited collectors: artwork and table browsing, physical copies, crates, shelf locations, search/barcode/photo capture, listening history, CSV export, printable record QR labels, selective crate sharing, and the Wax Wizard. Real collections start empty; fixtures belong to tests.

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

| Variable                | Purpose                                                                                                                                                  |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DATABASE_URL`          | Postgres URL for the current environment. Only production supports existing `neon_connect` as a fallback. Preview and development require their own URL. |
| `BETTER_AUTH_SECRET`    | Independent random secret of at least 32 characters per environment.                                                                                     |
| `BETTER_AUTH_URL`       | Canonical application origin. Use `http://localhost:3000` locally; preview can fall back to `VERCEL_URL`.                                                |
| `OPENAI_API_KEY`        | OpenAI API key. Existing `chaz_gpt` is supported as a fallback.                                                                                          |
| `WAX_AI_MODEL`          | Optional OpenAI model override; defaults to `gpt-5.4-mini`.                                                                                              |
| `BLOB_READ_WRITE_TOKEN` | Token for a private Blob store. A linked `BLOB_STORE_ID` with Vercel OIDC is also supported.                                                             |
| `DISCOGS_TOKEN`         | Optional Discogs personal-access token for authenticated lookup.                                                                                         |
| `WAX_OWNER_EMAIL`       | Email address bound to the initial owner invitation.                                                                                                     |
| `WAX_BOOTSTRAP_TOKEN`   | Random initial owner invitation token of at least 32 characters.                                                                                         |

Canonical environment names take precedence over their aliases. Runtime and migration checks require both the role and database to be `wax_preview` for `VERCEL_ENV=preview`, or `wax_development` for local/development processes. `VERCEL_ENV=production` selects production configuration. A missing or mismatched nonproduction URL fails before connection. Keep all values server-side and out of Git. `db:verify` checks database connectivity and reports whether AI and Blob configuration exists without printing credentials; it does not validate every provider's credentials.

## First owner and invitations

Set `WAX_OWNER_EMAIL` and a securely generated `WAX_BOOTSTRAP_TOKEN`, then run migrations for the intended environment. If no owner exists, the migration runner creates a one-use owner invitation valid for seven days.

If the unused owner invitation expires, rotate `WAX_BOOTSTRAP_TOKEN` to a new random value and run migrations again. Reusing the same token never extends its expiry; after an owner has registered, bootstrap configuration cannot replace that account.

Open `<app-origin>/join#token=<WAX_BOOTSTRAP_TOKEN>` privately, register using the configured email and a password of at least 12 characters, then sign in. The token is a URL fragment and only its hash is stored in Postgres. Treat the full link as a credential. After successful setup, remove the bootstrap token from deployment configuration.

The owner creates and revokes collector invitations in the application. Share those links with the intended recipients yourself; WAX does not send invitation emails. Public registration is disabled. Existing collectors receive their own private library.

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

Invitations currently provide access without an email-delivery service: email verification and self-service password recovery are not implemented. AI, Discogs, and Blob outages require retrying or manual entry. Uploaded photos are retained independently of copy deletion; a photo-retention/deletion interface is not yet available. Live provider access and physical iOS/Android camera behavior require deployment checks in addition to automated tests.

Listening-room tools, shop planning, commerce, public signup, marketplace valuation, and streaming playback are outside this release.
