import type { Metadata } from "next";
import { ShopsView } from "@/components/shops-view";

export const metadata: Metadata = { title: "Record shops nearby · WAX" };

export default function Page() {
  return <ShopsView />;
}
