import { requireCollector } from "@/server/auth";
import { handleApi, HttpError } from "@/server/http";
import { MAX_PHOTO_BYTES, uploadPhoto } from "@/server/photos";

export const runtime = "nodejs";
export const maxDuration = 60;
export async function POST(request: Request) {
  return handleApi(request, async () => {
    const collector = await requireCollector();
    if (
      Number(request.headers.get("content-length")) >
      MAX_PHOTO_BYTES + 64 * 1024
    )
      throw new HttpError(413, "Choose a photo smaller than 3 MB.");
    let data: FormData;
    try {
      data = await request.formData();
    } catch {
      throw new HttpError(400, "Choose a JPEG, PNG, or WebP photo to upload.");
    }
    const file = data.get("file");
    if (!(file instanceof File))
      throw new HttpError(400, "Choose a photo first.");
    return uploadPhoto(collector.id, file);
  });
}
