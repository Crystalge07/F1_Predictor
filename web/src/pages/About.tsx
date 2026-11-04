// About: how the predictions are made, kept honest, and scored. Plain words,
// for a visitor who doesn't know the project.

import { Link } from "react-router-dom";

export function About() {
  return (
    <section className="prose">
      <h1>How it works</h1>
      <p>
        A model predicts every Formula 1 Grand Prix, three times a weekend, and is scored in public after each race.
        Every prediction is saved before the session it predicts and never edited afterwards, so the scorecard can't be
        flattered.
      </p>

      <h2>Three predictions a weekend</h2>
      <ul>
        <li><strong>Pre-weekend</strong>, before the first practice: from history only (recent form, team pace, track record, the championship).</li>
        <li><strong>After practice</strong>, before qualifying: adds practice pace (long runs and best laps).</li>
        <li><strong>After qualifying</strong>, before the race: adds the official starting grid, the strongest single signal.</li>
      </ul>
      <p>
        Each stage uses only what was known at that moment: the cutoff is the start of the next session. A model
        (XGBoost) predicts where each driver finishes, a second one each driver's chance of retiring, and 10,000
        simulated races turn those into win, podium and points chances. The <Link to="/why">Why</Link> page breaks each
        prediction into the factors that moved it.
      </p>

      <h2>Kept honest</h2>
      <ul>
        <li>Predictions run on their own every hour (GitHub Actions) and are logged before each cutoff, with the exact code version that made them.</li>
        <li>The database refuses to change or delete a logged live prediction.</li>
        <li>After the race, every prediction is scored against a simple baseline that uses the same information: the championship order, the practice order, or the starting grid.</li>
        <li>
          The <Link to="/scorecard">Scorecard</Link> keeps three kinds of results apart: the 2027 season, predicted
          live (the real test); the last four races of 2026, predicted live as a rehearsal; and backtests, which replay
          past races after the fact and are only a check that the pipeline works.
        </li>
      </ul>

      <h2>Known limits</h2>
      <ul>
        <li>The pre-weekend prediction takes the driver list from the previous race, so it misses driver changes until practice.</li>
        <li>A driver making their debut has no history: they're predicted like a rookie, and can only be named once they've qualified.</li>
        <li>Weather forecasts aren't used yet.</li>
      </ul>

      <h2>Data</h2>
      <p>
        Results, qualifying and the calendar from <a href="https://github.com/jolpica/jolpica-f1">Jolpica-F1</a>; lap
        times, tyre stints, weather and the starting grid from <a href="https://openf1.org">OpenF1</a>. Not affiliated
        with Formula 1. Code: <a href="https://github.com/Crystalge07/F1_Predictor">github.com/Crystalge07/F1_Predictor</a>.
      </p>
    </section>
  );
}
