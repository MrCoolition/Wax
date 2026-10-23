import type { ShopAvatarId } from "@/lib/shop-notebook";

const palettes: Record<ShopAvatarId, [string, string, string]> = {
  groove: ["#d4ab46", "#9e583c", "#26251f"],
  sunshine: ["#d8a59a", "#e9b990", "#673f2c"],
  midnight: ["#9faeb6", "#87523e", "#282b39"],
  cherry: ["#c6553e", "#f0bf96", "#472c29"],
  mint: ["#aec3a1", "#b97850", "#343e31"],
  cosmic: ["#b6a3bd", "#e3a97d", "#402d44"],
};

/** Playful, chosen characters — never inferred portraits of real people. */
export function ShopAvatar({
  avatar,
  name,
}: {
  avatar: ShopAvatarId;
  name?: string;
}) {
  const [background, skin, hair] = palettes[avatar];
  return (
    <svg
      className="shop-avatar"
      viewBox="0 0 80 80"
      role={name ? "img" : undefined}
      aria-label={name ? `${name}’s avatar` : undefined}
      aria-hidden={name ? undefined : true}
    >
      <circle cx="40" cy="40" r="39" fill={background} />
      <path d="M11 80c1-20 13-28 29-28s28 8 29 28" fill={hair} />
      <path d="M33 48h14v14c-4 6-10 6-14 0" fill={skin} />
      <path
        d="M21 34c0-18 38-21 38 0v12c0 14-9 21-19 21S21 60 21 46Z"
        fill={skin}
      />
      {avatar === "groove" ? (
        <>
          <path
            d="M20 35V29c0-16 40-16 40 0v6c-7 1-13-6-14-12-6 10-17 10-26 12Z"
            fill={hair}
          />
          <path
            d="M16 38v-8c0-28 48-28 48 0v8"
            fill="none"
            stroke="#f7edce"
            strokeWidth="5"
          />
          <rect x="12" y="34" width="9" height="18" rx="4" fill="#f7edce" />
          <rect x="59" y="34" width="9" height="18" rx="4" fill="#f7edce" />
        </>
      ) : avatar === "sunshine" ? (
        <>
          <path
            d="M17 50c-9-10-7-30 3-31-1-13 13-18 21-10 11-10 23-1 21 9 12 4 13 23 3 32l-7-9V27c-9 5-22 4-33 0v15Z"
            fill={hair}
          />
          <circle
            cx="19"
            cy="51"
            r="5"
            fill="none"
            stroke="#f7df88"
            strokeWidth="3"
          />
          <circle
            cx="61"
            cy="51"
            r="5"
            fill="none"
            stroke="#f7df88"
            strokeWidth="3"
          />
        </>
      ) : avatar === "midnight" ? (
        <>
          <path d="M20 33C19 15 26 9 40 9s22 7 20 24Z" fill={hair} />
          <path d="M17 29h46v7H17Z" fill="#d8b75e" />
          <path d="M24 40h13v8H24Zm19 0h13v8H43Z" fill={hair} />
          <path d="M37 43h6" stroke={hair} strokeWidth="3" />
        </>
      ) : avatar === "cherry" ? (
        <>
          <path
            d="M17 59V29C17 3 64 4 64 31v28l-8-10V30c-8 0-14-5-17-10-3 9-10 13-16 13v17Z"
            fill={hair}
          />
          <path
            d="m22 24 6-9 26 1 5 9"
            fill="none"
            stroke="#f1d68b"
            strokeWidth="6"
          />
        </>
      ) : avatar === "mint" ? (
        <>
          <path
            d="M20 36c-4-16 5-24 20-24 17 0 24 8 20 24l-8-12c-9 8-17 10-29 9Z"
            fill={hair}
          />
          <circle
            cx="30"
            cy="42"
            r="8"
            fill="none"
            stroke="#f9f1d6"
            strokeWidth="3"
          />
          <circle
            cx="50"
            cy="42"
            r="8"
            fill="none"
            stroke="#f9f1d6"
            strokeWidth="3"
          />
          <path d="M38 41h4" stroke="#f9f1d6" strokeWidth="3" />
          <path d="M34 55q6-5 12 0" fill="none" stroke={hair} strokeWidth="3" />
        </>
      ) : (
        <>
          <path
            d="M17 37c-7-17 1-27 14-26 5-13 22-8 23 2 14 0 16 15 8 24l-5-9c-6 5-13 1-17-4-2 10-11 13-18 8Z"
            fill={hair}
          />
          <path
            d="m56 14 2 5 5 1-4 3 1 5-4-3-5 2 2-5-3-4 5 1Z"
            fill="#f5dc8d"
          />
          <circle cx="61" cy="52" r="3" fill="#f5dc8d" />
        </>
      )}
      {avatar !== "midnight" && (
        <g fill="#27241f">
          <ellipse cx="30" cy="42" rx="2" ry="2.5" />
          <ellipse cx="50" cy="42" rx="2" ry="2.5" />
        </g>
      )}
      <path
        d="M35 55q5 5 10 0"
        fill="none"
        stroke="#522f26"
        strokeWidth="2.3"
        strokeLinecap="round"
      />
      <path d="m25 68 8 12m22-12-8 12" stroke={background} strokeWidth="3" />
    </svg>
  );
}
