"use client";

import { useEffect, useMemo, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "";

type Match = {
  match_id: string;
  deliveries: number;
  season: string;
  team1?: string;
  team2?: string;
};

type Prediction = {
  delivery_number: number;
  score: number;
  wickets: number;
  win_probability: number;
  loss_probability: number;
  balls_bowled: number;
  balls_remaining: number;
  required_runs: number;
  current_run_rate: number;
  required_run_rate: number;
};

type ReplayResponse = {
  match_id: string;
  predictions: Prediction[];
};

export default function Home() {
  const [matches, setMatches] = useState<Match[]>([]);
  const [selectedMatch, setSelectedMatch] = useState("");

  const [replay, setReplay] = useState<Prediction[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);

  const [loadingMatches, setLoadingMatches] = useState(true);
  const [loadingReplay, setLoadingReplay] = useState(false);
  const [error, setError] = useState("");

  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);

  /*
   * Load available matches when the page opens.
   */
  useEffect(() => {
    async function loadMatches() {
      try {
        setLoadingMatches(true);
        setError("");

        const response = await fetch(
          `${API_URL}/api/replay/matches`
        );

        if (!response.ok) {
          throw new Error("Failed to load matches.");
        }

        const data = await response.json();

        setMatches(data.matches || []);

        if (data.matches?.length > 0) {
          setSelectedMatch(data.matches[0].match_id);
        }
      } catch (err) {
        console.error(err);
        setError(
          "Could not connect to the prediction server."
        );
      } finally {
        setLoadingMatches(false);
      }
    }

    loadMatches();
  }, []);

  /*
   * Load the complete replay whenever the selected match changes.
   */
  useEffect(() => {
    if (!selectedMatch) return;

    async function loadReplay() {
      try {
        setLoadingReplay(true);
        setError("");
        setIsPlaying(false);
        setCurrentIndex(0);

        const response = await fetch(
          `${API_URL}/api/replay/${selectedMatch}/series`
        );

        if (!response.ok) {
          throw new Error("Failed to load replay.");
        }

        const data: ReplayResponse = await response.json();

        setReplay(data.predictions || []);
      } catch (err) {
        console.error(err);
        setError(
          "Could not load the replay for this match."
        );
        setReplay([]);
      } finally {
        setLoadingReplay(false);
      }
    }

    loadReplay();
  }, [selectedMatch]);

  /*
   * Automatically advance through the replay.
   */
  useEffect(() => {
    if (!isPlaying || replay.length === 0) return;

    if (currentIndex >= replay.length - 1) {
      setIsPlaying(false);
      return;
    }

    const interval = setInterval(() => {
      setCurrentIndex((previous) => {
        if (previous >= replay.length - 1) {
          setIsPlaying(false);
          return previous;
        }

        return previous + 1;
      });
    }, 1000 / speed);

    return () => clearInterval(interval);
  }, [isPlaying, currentIndex, replay.length, speed]);

  const currentPrediction = replay[currentIndex];

  const selectedMatchInfo = useMemo(
    () =>
      matches.find(
        (match) => match.match_id === selectedMatch
      ),
    [matches, selectedMatch]
  );

  const probability =
    currentPrediction?.win_probability ?? 0;

  const probabilityPercentage = probability * 100;

  function previousDelivery() {
    setIsPlaying(false);

    setCurrentIndex((index) =>
      Math.max(0, index - 1)
    );
  }

  function nextDelivery() {
    setCurrentIndex((index) =>
      Math.min(replay.length - 1, index + 1)
    );
  }

  function togglePlay() {
    if (currentIndex >= replay.length - 1) {
      setCurrentIndex(0);
    }

    setIsPlaying((playing) => !playing);
  }

  return (
    <main className="min-h-screen bg-[#020617] text-white">
      <div className="mx-auto max-w-7xl px-5 py-8 md:px-8">

        {/* HEADER */}
        <header className="mb-8 flex flex-col gap-5 md:flex-row md:items-end md:justify-between">
          <div>
            <div className="mb-3 flex items-center gap-2">
              <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-emerald-400 text-lg text-slate-950">
                🏏
              </span>

              <span className="text-sm font-semibold uppercase tracking-[0.25em] text-emerald-400">
                IPL Analytics
              </span>
            </div>

            <h1 className="text-4xl font-bold tracking-tight md:text-5xl">
              Win Predictor
            </h1>

            <p className="mt-3 max-w-2xl text-slate-400">
              Replay historical IPL matches and watch the
              LSTM model update its win probability after
              every delivery.
            </p>
          </div>

          <div className="flex items-center gap-2 rounded-full border border-slate-800 bg-slate-900 px-4 py-2 text-sm">
            <span className="h-2 w-2 rounded-full bg-emerald-400" />
            <span className="text-slate-300">
              LSTM Model
            </span>
          </div>
        </header>

        {/* ERROR */}
        {error && (
          <div className="mb-6 rounded-2xl border border-red-900 bg-red-950/40 p-4 text-sm text-red-300">
            {error}
          </div>
        )}

        {/* MATCH SELECTOR */}
        <section className="mb-6 rounded-3xl border border-slate-800 bg-slate-900/80 p-5 shadow-2xl md:p-6">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-end">

            <div className="flex-1">
              <label className="mb-2 block text-sm font-medium text-slate-400">
                Historical Match
              </label>

              <select
                value={selectedMatch}
                onChange={(event) =>
                  setSelectedMatch(event.target.value)
                }
                disabled={loadingMatches}
                className="w-full appearance-none rounded-2xl border border-slate-700 bg-slate-950 px-5 py-4 text-base text-white outline-none transition focus:border-emerald-400"
              >
                {matches.map((match) => (
                  <option
                    key={match.match_id}
                    value={match.match_id}
                  >
                    Match {match.match_id} •{" "}
                    {match.season} •{" "}
                    {match.deliveries} deliveries
                  </option>
                ))}
              </select>
            </div>

            {selectedMatchInfo && (
              <div className="rounded-2xl border border-slate-800 bg-slate-950 px-5 py-4 text-sm">
                <p className="text-slate-500">
                  Match
                </p>

                <p className="mt-1 font-semibold">
                  {selectedMatchInfo.team1 ||
                    "Team 1"}{" "}
                  <span className="text-slate-600">
                    vs
                  </span>{" "}
                  {selectedMatchInfo.team2 ||
                    "Team 2"}
                </p>
              </div>
            )}
          </div>
        </section>

        {/* LOADING */}
        {loadingReplay && (
          <section className="rounded-3xl border border-slate-800 bg-slate-900 p-16 text-center">
            <div className="mx-auto mb-5 h-10 w-10 animate-spin rounded-full border-4 border-slate-700 border-t-emerald-400" />

            <p className="text-slate-300">
              Running the LSTM replay...
            </p>

            <p className="mt-2 text-sm text-slate-500">
              Calculating win probability across the innings.
            </p>
          </section>
        )}

        {/* DASHBOARD */}
        {!loadingReplay &&
          currentPrediction && (
            <>
              {/* SCORE + PROBABILITY */}
              <section className="grid gap-6 lg:grid-cols-[1.6fr_1fr]">

                {/* PROBABILITY */}
                <div className="relative overflow-hidden rounded-3xl border border-slate-800 bg-slate-900 p-7 md:p-10">
                  <div className="absolute -right-24 -top-24 h-64 w-64 rounded-full bg-emerald-400/5 blur-3xl" />

                  <div className="relative">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium uppercase tracking-[0.2em] text-slate-500">
                          Batting team
                        </p>

                        <h2 className="mt-2 text-xl font-semibold">
                          Win Probability
                        </h2>
                      </div>

                      <div className="rounded-full border border-slate-700 bg-slate-950 px-3 py-1 text-xs text-slate-400">
                        Delivery{" "}
                        {currentPrediction.delivery_number}
                      </div>
                    </div>

                    <div className="mt-10">
                      <span className="text-7xl font-bold tracking-tight text-emerald-400 md:text-8xl">
                        {probabilityPercentage.toFixed(
                          1
                        )}
                        %
                      </span>
                    </div>

                    {/* PROBABILITY BAR */}
                    <div className="mt-8">
                      <div className="mb-2 flex justify-between text-xs text-slate-500">
                        <span>Loss</span>
                        <span>Win</span>
                      </div>

                      <div className="h-4 overflow-hidden rounded-full bg-slate-800">
                        <div
                          className="h-full rounded-full bg-emerald-400 transition-all duration-500"
                          style={{
                            width: `${probabilityPercentage}%`,
                          }}
                        />
                      </div>
                    </div>

                    <p className="mt-5 text-sm text-slate-500">
                      Model confidence after{" "}
                      {currentPrediction.delivery_number}{" "}
                      deliveries.
                    </p>
                  </div>
                </div>

                {/* SCORECARD */}
                <div className="rounded-3xl border border-slate-800 bg-slate-900 p-7">
                  <p className="text-sm font-medium uppercase tracking-[0.2em] text-slate-500">
                    Current Score
                  </p>

                  <div className="mt-4">
                    <span className="text-6xl font-bold">
                      {currentPrediction.score}
                    </span>

                    <span className="ml-2 text-3xl text-slate-500">
                      /
                      {currentPrediction.wickets}
                    </span>
                  </div>

                  <p className="mt-2 text-sm text-slate-500">
                    After{" "}
                    {currentPrediction.balls_bowled}{" "}
                    legal balls
                  </p>

                  <div className="mt-8 grid grid-cols-2 gap-3">
                    <MiniStat
                      label="CRR"
                      value={currentPrediction.current_run_rate.toFixed(
                        2
                      )}
                    />

                    <MiniStat
                      label="RRR"
                      value={currentPrediction.required_run_rate.toFixed(
                        2
                      )}
                    />

                    <MiniStat
                      label="Required"
                      value={currentPrediction.required_runs.toString()}
                    />

                    <MiniStat
                      label="Balls Left"
                      value={currentPrediction.balls_remaining.toString()}
                    />
                  </div>
                </div>
              </section>

              {/* GRAPH */}
              <section className="mt-6 rounded-3xl border border-slate-800 bg-slate-900 p-6 md:p-8">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium uppercase tracking-[0.2em] text-slate-500">
                      Model Timeline
                    </p>

                    <h2 className="mt-1 text-xl font-semibold">
                      Win Probability
                    </h2>
                  </div>

                  <span className="text-sm text-slate-500">
                    {currentIndex + 1} /{" "}
                    {replay.length}
                  </span>
                </div>

                <ProbabilityChart
                  predictions={replay}
                  currentIndex={currentIndex}
                />
              </section>

              {/* CONTROLS */}
              <section className="mt-6 rounded-3xl border border-slate-800 bg-slate-900 p-6">
                <div className="flex flex-col gap-5">

                  {/* SLIDER */}
                  <input
                    type="range"
                    min={0}
                    max={Math.max(0, replay.length - 1)}
                    value={currentIndex}
                    onChange={(event) => {
                      setIsPlaying(false);
                      setCurrentIndex(
                        Number(event.target.value)
                      );
                    }}
                    className="w-full accent-emerald-400"
                  />

                  <div className="flex items-center justify-between text-xs text-slate-500">
                    <span>
                      Delivery{" "}
                      {currentPrediction.delivery_number}
                    </span>

                    <span>
                      {replay[replay.length - 1]
                        ?.delivery_number || 0}
                    </span>
                  </div>

                  {/* BUTTONS */}
                  <div className="flex flex-wrap items-center justify-center gap-3">

                    <button
                      onClick={previousDelivery}
                      disabled={currentIndex === 0}
                      className="rounded-xl border border-slate-700 bg-slate-950 px-5 py-3 text-sm font-medium transition hover:border-slate-500 disabled:cursor-not-allowed disabled:opacity-30"
                    >
                      ← Previous
                    </button>

                    <button
                      onClick={togglePlay}
                      className="rounded-xl bg-emerald-400 px-7 py-3 font-bold text-slate-950 transition hover:bg-emerald-300"
                    >
                      {isPlaying
                        ? "❚❚ Pause"
                        : "▶ Play"}
                    </button>

                    <button
                      onClick={nextDelivery}
                      disabled={
                        currentIndex ===
                        replay.length - 1
                      }
                      className="rounded-xl border border-slate-700 bg-slate-950 px-5 py-3 text-sm font-medium transition hover:border-slate-500 disabled:cursor-not-allowed disabled:opacity-30"
                    >
                      Next →
                    </button>

                    {/* SPEED */}
                    <div className="ml-2 flex rounded-xl border border-slate-700 bg-slate-950 p-1">
                      {[1, 2, 4].map(
                        (value) => (
                          <button
                            key={value}
                            onClick={() =>
                              setSpeed(value)
                            }
                            className={`rounded-lg px-3 py-2 text-xs font-semibold transition ${
                              speed === value
                                ? "bg-slate-700 text-white"
                                : "text-slate-500 hover:text-white"
                            }`}
                          >
                            {value}x
                          </button>
                        )
                      )}
                    </div>
                  </div>
                </div>
              </section>

              {/* CURRENT STATE */}
              <section className="mt-6 grid grid-cols-2 gap-4 md:grid-cols-4">
                <StatCard
                  label="Win Probability"
                  value={`${probabilityPercentage.toFixed(
                    1
                  )}%`}
                />

                <StatCard
                  label="Loss Probability"
                  value={`${(
                    currentPrediction.loss_probability *
                    100
                  ).toFixed(1)}%`}
                />

                <StatCard
                  label="Required Runs"
                  value={currentPrediction.required_runs.toString()}
                />

                <StatCard
                  label="Balls Remaining"
                  value={currentPrediction.balls_remaining.toString()}
                />
              </section>
            </>
          )}

        {/* FOOTER */}
        <footer className="mt-12 border-t border-slate-900 pt-6 text-center text-xs text-slate-600">
          IPL Win Predictor • LSTM-based historical match
          replay
        </footer>
      </div>
    </main>
  );
}


/* -------------------------------------------------------
   Small statistic card
------------------------------------------------------- */

function StatCard({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900 p-5">
      <p className="text-xs uppercase tracking-wider text-slate-500">
        {label}
      </p>

      <p className="mt-2 text-2xl font-bold">
        {value}
      </p>
    </div>
  );
}


/* -------------------------------------------------------
   Small scorecard statistic
------------------------------------------------------- */

function MiniStat({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-xl bg-slate-950 p-4">
      <p className="text-xs text-slate-500">
        {label}
      </p>

      <p className="mt-1 text-lg font-semibold">
        {value}
      </p>
    </div>
  );
}


/* -------------------------------------------------------
   Probability chart
------------------------------------------------------- */

function ProbabilityChart({
  predictions,
  currentIndex,
}: {
  predictions: Prediction[];
  currentIndex: number;
}) {
  if (predictions.length === 0) {
    return null;
  }

  const width = 1000;
  const height = 300;

  const paddingX = 50;
  const paddingY = 30;

  const chartWidth = width - paddingX * 2;
  const chartHeight = height - paddingY * 2;

  const points = predictions.map(
    (prediction, index) => {
      const x =
        paddingX +
        (index /
          Math.max(1, predictions.length - 1)) *
          chartWidth;

      const y =
        height -
        paddingY -
        prediction.win_probability *
          chartHeight;

      return {
        x,
        y,
      };
    }
  );

  const line = points
    .map(
      (point, index) =>
        `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`
    )
    .join(" ");

  const currentPoint = points[currentIndex];

  return (
    <div className="mt-8 overflow-hidden rounded-2xl bg-slate-950 p-3 md:p-5">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="h-auto w-full"
        preserveAspectRatio="none"
      >
        {/* Grid lines */}
        {[0, 25, 50, 75, 100].map(
          (value) => {
            const y =
              height -
              paddingY -
              (value / 100) *
                chartHeight;

            return (
              <g key={value}>
                <line
                  x1={paddingX}
                  x2={width - paddingX}
                  y1={y}
                  y2={y}
                  stroke="currentColor"
                  className="text-slate-800"
                  strokeWidth="1"
                />

                <text
                  x="8"
                  y={y + 4}
                  className="fill-slate-600 text-[12px]"
                >
                  {value}%
                </text>
              </g>
            );
          }
        )}

        {/* Probability line */}
        <path
          d={line}
          fill="none"
          stroke="currentColor"
          className="text-emerald-400"
          strokeWidth="4"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* Current position */}
        {currentPoint && (
          <>
            <line
              x1={currentPoint.x}
              x2={currentPoint.x}
              y1={paddingY}
              y2={height - paddingY}
              stroke="currentColor"
              className="text-slate-700"
              strokeDasharray="5 5"
            />

            <circle
              cx={currentPoint.x}
              cy={currentPoint.y}
              r="7"
              fill="currentColor"
              className="text-emerald-400"
            />

            <circle
              cx={currentPoint.x}
              cy={currentPoint.y}
              r="13"
              fill="none"
              stroke="currentColor"
              className="text-emerald-400/30"
              strokeWidth="2"
            />
          </>
        )}
      </svg>

      <div className="mt-2 flex justify-between px-8 text-xs text-slate-600">
        <span>
          Delivery{" "}
          {predictions[0]?.delivery_number}
        </span>

        <span>
          Delivery{" "}
          {
            predictions[predictions.length - 1]
              ?.delivery_number
          }
        </span>
      </div>
    </div>
  );
}