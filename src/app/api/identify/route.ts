import { z } from "zod";
import { requireCollector } from "@/server/auth";
import { handleApi, HttpError } from "@/server/http";
import { identifyPhoto } from "@/server/photos";

export const runtime = "nodejs";
export const maxDuration = 90;
export async function POST(request: Request) {
  return handleApi(request, async () => {
    const collector = await requireCollector();
    if (Number(request.headers.get("content-length")) > 4096)
      throw new HttpError(413, "This request is too large.");
    const { photoId } = z
      .object({ photoId: z.string().uuid() })
      .strict()
      .parse(await request.json());
    return identifyPhoto(collector.id, photoId);
  });
}
