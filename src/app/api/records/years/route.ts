import { requireCollector } from "@/server/auth";
import { handleApi, HttpError } from "@/server/http";
import { consumeIntegrationLimit } from "@/server/integration-common";
import {
  enrichOriginalYears,
  originalYearRequestSchema,
} from "@/server/original-years";

export const runtime = "nodejs";
export const maxDuration = 60;

export async function POST(request: Request) {
  const response = await handleApi(request, async () => {
    const collector = await requireCollector();
    // The endpoint accepts no owner/copy/provider identifiers from the client.
    if (Number(request.headers.get("content-length")) > 128)
      throw new HttpError(400, "This request does not accept record data.");
    const reader = request.body?.getReader();
    if (!reader) throw new HttpError(400, "Send an empty request object.");
    const chunks: Uint8Array[] = [];
    let length = 0;
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        length += value.byteLength;
        if (length > 128) {
          await reader.cancel();
          throw new HttpError(400, "This request does not accept record data.");
        }
        chunks.push(value);
      }
    } finally {
      reader.releaseLock();
    }
    originalYearRequestSchema.parse(
      JSON.parse(Buffer.concat(chunks, length).toString("utf8")),
    );
    await consumeIntegrationLimit(collector.id, "original-years", 10, 60);
    return enrichOriginalYears(collector.id);
  });
  response.headers.set("Cache-Control", "private, no-store");
  return response;
}
