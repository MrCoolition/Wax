import { requireCollector } from "@/server/auth";
import { getLibrary } from "@/server/collection";
import { handleApi } from "@/server/http";
export const dynamic = "force-dynamic";
export async function GET(request: Request) {
  return handleApi(request, async () =>
    getLibrary((await requireCollector()).id),
  );
}
