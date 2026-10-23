import { z } from "zod";
import { requireCollector } from "@/server/auth";
import { fetchRelease, safeDiscogsImage } from "@/server/catalog";
import { handleApi, HttpError } from "@/server/http";
import { consumeIntegrationLimit } from "@/server/integration-common";

export const runtime = "nodejs";

export async function GET(
  request: Request,
  context: { params: Promise<{ id: string }> },
) {
  const response = await handleApi(request, async () => {
    const collector = await requireCollector();
    const { id } = await context.params;
    const releaseId = z
      .string()
      .max(16)
      .regex(/^[1-9]\d*$/)
      .transform(Number)
      .pipe(z.number().int().positive().max(Number.MAX_SAFE_INTEGER))
      .parse(id);
    await consumeIntegrationLimit(collector.id, "catalog-artwork", 240, 60);
    const release = await fetchRelease(releaseId);
    const image = safeDiscogsImage(release.coverUrl);
    if (!image)
      throw new HttpError(404, "No cover image is listed for this release.");
    // The browser fetches only a validated public CDN image. WAX does not proxy
    // arbitrary URLs or send collector credentials to the image host.
    return new Response(null, { status: 302, headers: { Location: image } });
  });
  response.headers.set("Cache-Control", "private, no-store");
  response.headers.set("Vary", "Cookie");
  response.headers.set("Referrer-Policy", "no-referrer");
  return response;
}
