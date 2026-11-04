// What a page shows while its data loads, or if it can't be loaded.

export function Loading({ what = "data" }: { what?: string }) {
  return (
    <p className="status" role="status">
      Loading {what}…
    </p>
  );
}

export function ErrorBox({ message }: { message: string }) {
  return (
    <div className="error" role="alert">
      <strong>The data couldn't be shown.</strong>
      <p>{message}</p>
    </div>
  );
}
