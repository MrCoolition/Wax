export function parseConnectionString(value: string | undefined): string {
  const input = value?.trim();
  if (!input) throw new Error("Database connection is not configured.");
  const wrapped = input.match(
    /^psql\s+(['"])(postgres(?:ql)?:\/\/.*?)\1(?:\s+.*)?$/s,
  );
  const uri = wrapped ? wrapped[2] : input.replace(/^(['"])(.*)\1$/s, "$2");
  let url: URL;
  try {
    url = new URL(uri);
  } catch {
    throw new Error("Database connection is not a valid Postgres URL.");
  }
  if (!["postgres:", "postgresql:"].includes(url.protocol))
    throw new Error("Database connection must be a Postgres URL.");
  return uri;
}

/** Preview and local processes must never fall back to the production connection. */
export function databaseUrl(
  environment: Partial<NodeJS.ProcessEnv> = process.env,
): string {
  const target =
    environment.VERCEL_ENV === "production"
      ? "production"
      : environment.VERCEL_ENV === "preview"
        ? "wax_preview"
        : "wax_development";
  const connection = parseConnectionString(
    target === "production"
      ? environment.DATABASE_URL || environment.neon_connect
      : environment.DATABASE_URL,
  );
  if (target !== "production") {
    const url = new URL(connection);
    // node-postgres permits the query-string user to override the authority username.
    const role =
      url.searchParams.get("user") || decodeURIComponent(url.username);
    if (
      role !== target ||
      decodeURIComponent(url.pathname.slice(1)) !== target
    ) {
      throw new Error(
        `Database isolation check failed: this environment requires the ${target} database and role.`,
      );
    }
  }
  return connection;
}
