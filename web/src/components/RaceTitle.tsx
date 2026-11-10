// A race name as the page's big headline: "Las Vegas *Grand Prix*", with
// "Grand Prix" in the display face's italic (the editorial look of the
// design). Screen readers and tests still read the plain name.

export function RaceTitle({ name }: { name: string }) {
  const suffix = " Grand Prix";
  if (!name.endsWith(suffix)) return <h1 className="display">{name}</h1>;
  return (
    <h1 className="display">
      {name.slice(0, -suffix.length)} <em>Grand Prix</em>
    </h1>
  );
}
