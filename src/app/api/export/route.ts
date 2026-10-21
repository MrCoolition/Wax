import { requireCollector } from "@/server/auth";
import { exportCollection } from "@/server/collection";
import { handleApi } from "@/server/http";
export const dynamic = "force-dynamic";
export async function GET(request: Request) {
  return handleApi(request, async () =>
    exportCollection((await requireCollector()).id),
  );
}
