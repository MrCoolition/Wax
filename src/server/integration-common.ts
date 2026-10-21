import { createOpenAI } from "@ai-sdk/openai";
import { getPool } from "./db";
import { HttpError } from "./http";

export function getAiModel() {
  const apiKey = process.env.OPENAI_API_KEY || process.env.chaz_gpt;
  if (!apiKey)
    throw new HttpError(
      503,
      "The Wizard is not connected yet. Your collection is safe; try again later.",
    );
  const model = (process.env.WAX_AI_MODEL || "gpt-5.4-mini").replace(
    /^openai\//,
    "",
  );
  return createOpenAI({ apiKey }).responses(model);
}

// A database-backed fixed window works across concurrent Vercel instances.
export async function consumeIntegrationLimit(
  ownerId: string,
  operation: string,
  limit: number,
  seconds = 3600,
) {
  const result = await getPool().query<{ count: number; retry_after: number }>(
    `
    INSERT INTO wax.integration_limits(owner_id,operation,window_start,count)
    VALUES($1,$2,now(),1)
    ON CONFLICT(owner_id,operation) DO UPDATE SET
      count = CASE WHEN wax.integration_limits.window_start < now()-make_interval(secs => $3)
        THEN 1 ELSE wax.integration_limits.count+1 END,
      window_start = CASE WHEN wax.integration_limits.window_start < now()-make_interval(secs => $3)
        THEN now() ELSE wax.integration_limits.window_start END
    RETURNING count, greatest(1,ceil(extract(epoch FROM window_start+make_interval(secs => $3)-now())))::int AS retry_after
  `,
    [ownerId, operation, seconds],
  );
  if (result.rows[0].count > limit)
    throw new HttpError(
      429,
      "A lot of digging at once! Please wait before trying again.",
      { retryAfter: result.rows[0].retry_after },
    );
}

export function aiUnavailable(error: unknown, message: string): never {
  if (error instanceof HttpError) throw error;
  const status =
    typeof error === "object" && error !== null && "statusCode" in error
      ? error.statusCode
      : null;
  throw new HttpError(
    status === 429 ? 429 : 503,
    message,
    status === 429 ? { retryAfter: 60 } : undefined,
  );
}
