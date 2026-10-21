"use client";
import { useEffect, useRef, useState, type FormEvent } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  ArrowLeft,
  ArrowRight,
  Camera,
  Check,
  Disc3,
  ImagePlus,
  LoaderCircle,
  PencilLine,
  ScanLine,
  Search,
  X,
} from "lucide-react";
import type {
  CatalogCandidate,
  CopyFields,
  PhotoIdentification,
  RecordCopy,
} from "@/lib/contracts";
import { DEFAULT_COPY, EMPTY_CANDIDATE } from "@/lib/contracts";
import { api, ApiError, useLibrary } from "./library-provider";
import {
  Button,
  Cover,
  ErrorNotice,
  Field,
  PageHeader,
  SourceCredit,
} from "./ui";
import { CandidateFields, CopyForm } from "./record-form";

type Mode = "search" | "scan" | "photo" | "manual";
async function compressImage(file: File): Promise<File> {
  if (!file.type.startsWith("image/"))
    throw new Error("Choose a photo in JPEG, PNG, or WebP format.");
  const bitmap = await createImageBitmap(file).catch(() => {
    throw new Error(
      "That image format could not be opened. Try a JPEG or PNG photo.",
    );
  });
  const canvas = document.createElement("canvas");
  const scale = Math.min(1, 1800 / Math.max(bitmap.width, bitmap.height));
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  canvas.getContext("2d")!.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close();
  const blob = await new Promise<Blob>((resolve, reject) =>
    canvas.toBlob(
      (value) =>
        value
          ? resolve(value)
          : reject(new Error("Could not prepare that photo.")),
      "image/jpeg",
      0.88,
    ),
  );
  if (blob.size > 3 * 1024 * 1024)
    throw new Error("Please choose a smaller photo, under 3 MB.");
  return new File([blob], "record-photo.jpg", { type: "image/jpeg" });
}
function BarcodeCamera({
  onResult,
  onClose,
}: {
  onResult: (value: string) => void;
  onClose: () => void;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const resultHandler = useRef(onResult);
  resultHandler.current = onResult;
  const [error, setError] = useState("");
  useEffect(() => {
    let stopped = false;
    let controls: { stop: () => void } | undefined;
    const currentVideo = video.current;
    void import("@zxing/browser")
      .then(async ({ BrowserMultiFormatOneDReader }) => {
        if (stopped || !currentVideo) return;
        const reader = new BrowserMultiFormatOneDReader();
        const active = await reader.decodeFromConstraints(
          { audio: false, video: { facingMode: { ideal: "environment" } } },
          currentVideo,
          (result, _, scanner) => {
            if (result && !stopped) {
              stopped = true;
              scanner.stop();
              resultHandler.current(result.getText());
            }
          },
        );
        controls = active;
        if (stopped) active.stop();
      })
      .catch((err) => {
        if (!stopped)
          setError(
            err?.name === "NotAllowedError"
              ? "Camera access was denied. Allow camera access in your browser, or enter the barcode below."
              : "We couldn’t start that camera. You can still type the barcode below.",
          );
      });
    return () => {
      stopped = true;
      controls?.stop();
      const stream = currentVideo?.srcObject;
      if (stream instanceof MediaStream)
        stream.getTracks().forEach((track) => track.stop());
    };
  }, []);
  return (
    <div className="camera-box">
      <div className="camera-top">
        <span>
          <ScanLine size={17} /> FIND THE BARCODE
        </span>
        <button
          className="icon-button"
          onClick={onClose}
          aria-label="Stop camera"
        >
          <X size={20} />
        </button>
      </div>
      <video ref={video} muted playsInline autoPlay />
      <div className="scan-frame" aria-hidden="true" />
      <p>Keep the barcode flat and steady in the frame.</p>
      <ErrorNotice>{error}</ErrorNotice>
    </div>
  );
}
export function AddView() {
  const params = useSearchParams();
  const router = useRouter();
  const { refresh, data } = useLibrary();
  const initialMode = params.get("mode");
  const [mode, setMode] = useState<Mode>(
    ["scan", "photo", "manual"].includes(initialMode || "")
      ? (initialMode as Mode)
      : "search",
  );
  const [query, setQuery] = useState("");
  const [searchType, setSearchType] = useState("query");
  const [barcode, setBarcode] = useState("");
  const [camera, setCamera] = useState(false);
  const [candidates, setCandidates] = useState<CatalogCandidate[]>([]);
  const [searched, setSearched] = useState(false);
  const [candidate, setCandidate] = useState<CatalogCandidate | null>(null);
  const [copy, setCopy] = useState<CopyFields>({ ...DEFAULT_COPY });
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [clue, setClue] = useState("");
  const [photoId, setPhotoId] = useState<string | null>(null);
  const [photo, setPhoto] = useState<File | null>(null);
  const [photoPreview, setPhotoPreview] = useState("");
  const [allowDuplicate, setAllowDuplicate] = useState(false);
  const [duplicate, setDuplicate] = useState(false);
  const [crateIds, setCrateIds] = useState<string[]>([]);
  const requestKey = useRef<string | null>(null);
  useEffect(() => {
    if (!photo) {
      setPhotoPreview("");
      return;
    }
    const url = URL.createObjectURL(photo);
    setPhotoPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [photo]);
  async function search(params: Record<string, string>) {
    setBusy("search");
    setError("");
    setSearched(false);
    try {
      const result = await api<{ candidates: CatalogCandidate[] }>(
        "/api/catalog?" + new URLSearchParams(params),
      );
      setCandidates(result.candidates);
      setSearched(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search unavailable.");
    } finally {
      setBusy("");
    }
  }
  async function selectCandidate(item: CatalogCandidate) {
    setBusy("release");
    setError("");
    try {
      const full = item.discogsId
        ? await api<CatalogCandidate>("/api/catalog/" + item.discogsId)
        : item;
      setCandidate(full);
      setCopy({ ...DEFAULT_COPY });
      setDuplicate(false);
      setAllowDuplicate(false);
      requestKey.current = null;
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Could not load that pressing.",
      );
    } finally {
      setBusy("");
    }
  }
  async function identify() {
    if (!photo) return;
    setBusy("photo");
    setError("");
    try {
      let currentPhotoId = photoId;
      if (!currentPhotoId) {
        const prepared = await compressImage(photo);
        const form = new FormData();
        form.append("file", prepared);
        const uploaded = await api<{ id: string; url: string }>("/api/photos", {
          method: "POST",
          body: form,
        });
        currentPhotoId = uploaded.id;
        setPhotoId(currentPhotoId);
      }
      const result = await api<PhotoIdentification>("/api/identify", {
        method: "POST",
        body: JSON.stringify({ photoId: currentPhotoId }),
      });
      setCandidates(result.candidates);
      setSearched(true);
      setClue(
        [result.clues.artist, result.clues.title, result.clues.uncertainty]
          .filter(Boolean)
          .join(" · "),
      );
      if (!result.candidates.length)
        setCandidate({
          ...EMPTY_CANDIDATE,
          artist: result.clues.artist || "",
          title: result.clues.title || "",
          label: result.clues.label || "",
          catalogNumber: result.clues.catalogNumber || "",
        });
    } catch (err) {
      setError(
        (err instanceof Error
          ? err.message
          : "Could not identify that photo.") +
          " Your photo is still here. Retry, or choose Add by hand to complete the details.",
      );
    } finally {
      setBusy("");
    }
  }
  async function save(event: FormEvent) {
    event.preventDefault();
    if (!candidate) return;
    setBusy("save");
    setError("");
    requestKey.current ||= crypto.randomUUID();
    try {
      const result = await api<RecordCopy>("/api/records", {
        method: "POST",
        body: JSON.stringify({
          idempotencyKey: requestKey.current,
          candidate: { ...candidate, genres: candidate.genres.filter(Boolean) },
          copy,
          allowDuplicate,
          photoIds: photoId ? [photoId] : [],
          crateIds,
        }),
      });
      await refresh();
      router.push("/records/" + result.id);
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        setDuplicate(true);
        requestKey.current = null;
      }
      setError(
        err instanceof Error ? err.message : "Could not save your record.",
      );
    } finally {
      setBusy("");
    }
  }
  const existing =
    candidate &&
    data?.records.some((r) =>
      candidate.discogsId
        ? r.discogsId === candidate.discogsId
        : r.title.toLowerCase() === candidate.title.toLowerCase() &&
          r.artist.toLowerCase() === candidate.artist.toLowerCase(),
    );
  return (
    <>
      <PageHeader
        eyebrow="NEW ARRIVALS / MAKE ROOM ON THE SHELF"
        title="Add some wax"
        description="Find the release. Check the pressing. Make it yours."
      />
      <div className="flow-steps">
        <span className={!candidate ? "current" : "complete"}>
          <b>{candidate ? <Check size={13} /> : "1"}</b> Find your record
        </span>
        <i />
        <span className={candidate ? "current" : ""}>
          <b>2</b> The pressing & your copy
        </span>
        <i />
        <span>
          <b>3</b> Into the collection
        </span>
      </div>
      {candidate ? (
        <form onSubmit={save} className="review-layout">
          <div className="review-main">
            <button
              type="button"
              className="text-button"
              onClick={() => {
                setCandidate(null);
                setError("");
              }}
            >
              <ArrowLeft size={15} /> Back to finding your record
            </button>
            <div className="section-title">
              <span>01 / THE RELEASE</span>
              <h2>Let’s get the details right.</h2>
            </div>
            <CandidateFields value={candidate} onChange={setCandidate} />
            <div className="section-title">
              <span>02 / YOUR COPY</span>
              <h2>Same record. Your story.</h2>
            </div>
            <CopyForm value={copy} onChange={setCopy} />
            {Boolean(data?.crates.length) && (
              <fieldset className="crate-checkboxes">
                <legend>File it in a crate</legend>
                {data!.crates.map((crate) => (
                  <label key={crate.id}>
                    <input
                      type="checkbox"
                      checked={crateIds.includes(crate.id)}
                      onChange={(e) =>
                        setCrateIds(
                          e.target.checked
                            ? [...crateIds, crate.id]
                            : crateIds.filter((id) => id !== crate.id),
                        )
                      }
                    />
                    <i style={{ background: crate.color }} />
                    {crate.name}
                  </label>
                ))}
              </fieldset>
            )}
            {(existing || duplicate) && (
              <label className="checkbox-row duplicate-box">
                <input
                  type="checkbox"
                  checked={allowDuplicate}
                  onChange={(e) => setAllowDuplicate(e.target.checked)}
                />
                <span>
                  <strong>Looks like this one’s already on your shelf.</strong>
                  <small>
                    Check this to add another physical copy. Your existing copy
                    stays as it is.
                  </small>
                </span>
              </label>
            )}
            <ErrorNotice>{error}</ErrorNotice>
            <div className="form-actions">
              <Button
                type="submit"
                busy={busy === "save"}
                disabled={Boolean((existing || duplicate) && !allowDuplicate)}
              >
                Add to my collection <ArrowRight size={17} />
              </Button>
              <span>Saved privately. Always yours.</span>
            </div>
          </div>
          <aside className="review-sleeve">
            <Cover
              record={{
                ...candidate,
                coverUrl: photoPreview || candidate.coverUrl,
              }}
            />
            <span className="eyebrow">YOUR NEXT GOOD FIND</span>
            <h3>{candidate.title || "Untitled record"}</h3>
            <p>{candidate.artist || "Unknown artist"}</p>
            <div className="pressing-status">
              <span
                className={
                  "status-dot " + (copy.pressingConfirmed ? "green" : "")
                }
              />
              {copy.pressingConfirmed
                ? "Pressing checked"
                : "Pressing unconfirmed"}
            </div>
            <SourceCredit sourceUrl={candidate.sourceUrl} />
            <p className="small-note">
              A cover can tell us the album. The label, catalog number, and
              runout tell us the pressing.
            </p>
          </aside>
        </form>
      ) : (
        <>
          <div className="capture-tabs">
            {[
              { id: "search", label: "Search the catalog", icon: Search },
              { id: "scan", label: "Scan a barcode", icon: ScanLine },
              { id: "photo", label: "Snap a photo", icon: Camera },
              { id: "manual", label: "Add by hand", icon: PencilLine },
            ].map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                className={mode === id ? "active" : ""}
                onClick={() => {
                  setMode(id as Mode);
                  setCamera(false);
                  setError("");
                  setSearched(false);
                  setClue("");
                }}
              >
                <Icon size={20} />
                <span>{label}</span>
              </button>
            ))}
          </div>
          <section className="capture-panel">
            {mode === "search" && (
              <>
                <span className="eyebrow">THE DIG STARTS HERE</span>
                <h2>What are we looking for?</h2>
                <p>
                  Search an artist, a record title, or the numbers on the spine.
                </p>
                <form
                  className="catalog-search"
                  onSubmit={(e) => {
                    e.preventDefault();
                    void search({ [searchType]: query });
                  }}
                >
                  <select
                    aria-label="Catalog search type"
                    value={searchType}
                    onChange={(e) => setSearchType(e.target.value)}
                  >
                    <option value="query">Everything</option>
                    <option value="artist">Artist</option>
                    <option value="title">Title</option>
                    <option value="catalogNumber">Catalog no.</option>
                  </select>
                  <input
                    required
                    aria-label="Search the record catalog"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Artist, album, or catalog number…"
                  />
                  <Button type="submit" busy={busy === "search"}>
                    <Search size={17} /> Search
                  </Button>
                </form>
              </>
            )}
            {mode === "scan" && (
              <>
                <span className="eyebrow">A QUICK FLIP TO THE BACK</span>
                <h2>Scan it. Then check it.</h2>
                <p>
                  A barcode can match several pressings. You’ll choose the right
                  one next.
                </p>
                {camera ? (
                  <BarcodeCamera
                    onClose={() => setCamera(false)}
                    onResult={(value) => {
                      setBarcode(value);
                      setCamera(false);
                      void search({ barcode: value });
                    }}
                  />
                ) : (
                  <Button onClick={() => setCamera(true)}>
                    <Camera size={18} /> Open camera
                  </Button>
                )}
                <form
                  className="barcode-manual"
                  onSubmit={(e) => {
                    e.preventDefault();
                    void search({ barcode });
                  }}
                >
                  <Field label="Or type the barcode">
                    <input
                      value={barcode}
                      inputMode="numeric"
                      required
                      onChange={(e) => setBarcode(e.target.value)}
                      placeholder="Keep any leading zeros"
                    />
                  </Field>
                  <Button
                    variant="secondary"
                    type="submit"
                    busy={busy === "search"}
                  >
                    Find matches <ArrowRight size={16} />
                  </Button>
                </form>
              </>
            )}
            {mode === "photo" && (
              <>
                <span className="eyebrow">LET THE SLEEVE DO THE TALKING</span>
                <h2>Recognize a record.</h2>
                <p>
                  Photograph the cover or label in good light. You’ll review the
                  matches before anything is saved.
                </p>
                <div className="photo-capture">
                  <label
                    className={
                      "photo-drop " + (photoPreview ? "has-photo" : "")
                    }
                  >
                    {photoPreview ? (
                      <img
                        src={photoPreview}
                        alt="Your selected record photo"
                      />
                    ) : (
                      <>
                        <ImagePlus size={35} />
                        <strong>Choose a photo or open your camera</strong>
                        <span>JPEG, PNG, or WebP · we’ll size it for you</span>
                      </>
                    )}
                    <input
                      type="file"
                      accept="image/jpeg,image/png,image/webp"
                      capture="environment"
                      onChange={(e) => {
                        setPhoto(e.target.files?.[0] || null);
                        setPhotoId(null);
                        setSearched(false);
                      }}
                    />
                  </label>
                  {photo && (
                    <Button
                      onClick={() => void identify()}
                      busy={busy === "photo"}
                    >
                      <Camera size={17} />{" "}
                      {busy === "photo"
                        ? "Looking closely…"
                        : "Identify this record"}
                    </Button>
                  )}
                </div>
              </>
            )}
            {mode === "manual" && (
              <>
                <span className="eyebrow">YOU KNOW YOUR RECORDS</span>
                <h2>Keep it simple.</h2>
                <p>
                  Start with the artist and title. Add the finer details now, or
                  come back when you’ve had a closer look.
                </p>
                <Button
                  onClick={() => {
                    setCandidate({ ...EMPTY_CANDIDATE });
                    setCopy({ ...DEFAULT_COPY });
                  }}
                >
                  Enter the details <PencilLine size={17} />
                </Button>
              </>
            )}
            <ErrorNotice>{error}</ErrorNotice>
          </section>
          {clue && <div className="notice">Photo clues: {clue}</div>}
          {busy === "release" && (
            <div className="inline-loading" role="status">
              <LoaderCircle className="spin" size={20} /> Pulling the pressing
              details…
            </div>
          )}
          {searched && (
            <section className="candidate-results">
              <div className="section-title result-title">
                <div>
                  <span>THE SEARCH PARTY</span>
                  <h2>
                    {candidates.length
                      ? candidates.length + " possible matches."
                      : "No clear match this time."}
                  </h2>
                </div>
                <Button
                  variant="quiet"
                  onClick={() => setCandidate({ ...EMPTY_CANDIDATE })}
                >
                  Add by hand <PencilLine size={15} />
                </Button>
              </div>
              {!candidates.length && (
                <p>
                  Try the label or catalog number, take a clearer photo, or
                  enter what you know.
                </p>
              )}
              <div className="candidate-grid">
                {candidates.map((item, index) => (
                  <article
                    className="candidate-card"
                    key={item.discogsId || index}
                  >
                    <Cover record={item} />
                    <div>
                      <h3>{item.title}</h3>
                      <p>{item.artist}</p>
                      <small>
                        {[
                          item.year,
                          item.country,
                          item.label,
                          item.catalogNumber,
                          item.format,
                        ]
                          .filter(Boolean)
                          .join(" · ")}
                      </small>
                      <SourceCredit sourceUrl={item.sourceUrl} />
                      <Button
                        variant="secondary"
                        busy={busy === "release"}
                        onClick={() => void selectCandidate(item)}
                      >
                        Review this pressing <ArrowRight size={14} />
                      </Button>
                    </div>
                  </article>
                ))}
              </div>
            </section>
          )}
          <div className="capture-footnote">
            <Disc3 size={20} />
            <p>
              Different pressing? Different story. We’ll keep your physical
              copies separate, even when the album is the same.
            </p>
          </div>
        </>
      )}
    </>
  );
}
