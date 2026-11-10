// About: how the predictions are made, kept honest, and scored. Plain words,
// for a visitor who doesn't know the project.
//
// Laid out as a grid of cards so it fills the screen like the other pages:
//
//   intro (eyebrow, headline, one paragraph)
//   [ 01 Pre-weekend ] [ 02 After practice ] [ 03 After qualifying ]
//   [ The model ]      [ Kept honest ]       [ Known limits ]
//   [ Data ......................................................... ]
//
// The grids are CSS auto-fit (.about-grid), so they fall to two columns,
// then one, as the screen narrows.

import { Link } from "react-router-dom";

// The three stages, in weekend order: when it's made, and what it adds.
const STAGES = [
  {
    name: "Pre-weekend",
    when: "Before the first practice",
    what: "From history only: recent form, team pace, track record, the championship.",
  },
  {
    name: "After practice",
    when: "Before qualifying",
    what: "Adds practice pace: long runs and best laps.",
  },
  {
    name: "After qualifying",
    when: "Before the race",
    what: "Adds the official starting grid, the strongest single signal.",
  },
];

export function About() {
  return (
    <section className="about">
      <div className="about-intro">
        <p className="eyebrow">About the project</p>
        <h1>How it works</h1>
        <p className="lead">
          A model predicts every Formula 1 Grand Prix, three times a weekend, and is scored in public after each race.
          Every prediction is saved before the session it predicts and never edited afterwards, so the scorecard can't be
          flattered.
        </p>
      </div>

      <h2 className="section-label">Three predictions a weekend</h2>
      <ol className="about-grid stage-cards">
        {STAGES.map((stage, i) => (
          <li key={stage.name} className="card stage-card">
            {/* "01", "02", "03": the step number, decorative (the list already numbers them). */}
            <span className="stage-num" aria-hidden="true">{String(i + 1).padStart(2, "0")}</span>
            <h3>{stage.name}</h3>
            <p className="stage-when">{stage.when}</p>
            <p>{stage.what}</p>
          </li>
        ))}
      </ol>

      <div className="about-grid">
        <div className="card">
          <h2>The model</h2>
          <p>
            Each stage uses only what was known at that moment: the cutoff is the start of the next session. A model
            (XGBoost) predicts where each driver finishes, a second one each driver's chance of retiring, and 10,000
            simulated races turn those into win, podium and points chances.
          </p>
          <p>
            Click any driver in a prediction to see <Link to="/why">why</Link>: the factors that moved their prediction.
          </p>
        </div>

        <div className="card">
          <h2>Kept honest</h2>
          <ul>
            <li>Predictions run on their own every hour (GitHub Actions) and are logged before each cutoff, with the exact code version that made them.</li>
            <li>The database refuses to change or delete a logged live prediction.</li>
            <li>After the race, every prediction is scored against a simple baseline that uses the same information: the championship order, the practice order, or the starting grid.</li>
            <li>
              The <Link to="/results/stats">Results stats</Link> keep three kinds of results apart: the 2027 season, predicted
              live (the real test); the last four races of 2026, predicted live as a rehearsal; and backtests, which replay
              past races after the fact and are only a check that the pipeline works.
            </li>
          </ul>
        </div>

        <div className="card">
          <h2>Known limits</h2>
          <ul>
            <li>The pre-weekend prediction takes the driver list from the previous race, so it misses driver changes until practice.</li>
            <li>A driver making their debut has no history: they're predicted like a rookie, and can only be named once they've qualified.</li>
            <li>Weather forecasts aren't used yet.</li>
          </ul>
        </div>
      </div>

      <div className="card about-data">
        <h2>Data</h2>
        <p>
          Results, qualifying and the calendar from <a href="https://github.com/jolpica/jolpica-f1">Jolpica-F1</a>; lap
          times, tyre stints, weather and the starting grid from <a href="https://openf1.org">OpenF1</a>. Not affiliated
          with Formula 1. Code: <a href="https://github.com/Crystalge07/F1_Predictor">github.com/Crystalge07/F1_Predictor</a>.
        </p>
      </div>
    </section>
  );
}
