import { getAuth } from "@/server/auth";
import { handleApi } from "@/server/http";
export async function GET(request: Request) {
  return getAuth().handler(request);
}
export async function POST(request: Request) {
  return handleApi(request, () => getAuth().handler(request));
}
