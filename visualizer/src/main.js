
import './style.css';
import Chart from 'chart.js/auto';
import { loadExperiment } from './utils/dataloader.js';
import { analyzeExperiments } from './utils/experimentAnalysis.js';

// =============================================
// 1. EXPERIMENT CONFIGURATION
// =============================================

const experiments = [
  { id: 'E1', name: 'Fixed Batch + Fixed LR', color: '#60a5fa' },
  { id: 'E2', name: 'Adaptive Batch + Fixed LR', color: '#34d399' },
  { id: 'E3', name: 'Fixed Batch + Adaptive LR', color: '#c084fc' },
  { id: 'E4', name: 'Adaptive Batch + Adaptive LR', color: '#fbbf24' }
];

// =============================================
// 2. BUILD THE PAGE
// =============================================

document.querySelector('#app').innerHTML = `
  <div class="dashboard">

    <header class="header">
      <div>
        <h1>Adaptive CNN Visualizer</h1>
        <p>E1–E4 Optimization Strategy Comparison</p>
      </div>
      <span class="header-tag">Recorded Training Replay</span>
    </header>

    <section class="comparison-toolbar">

      <label>
        Dataset
        <select id="datasetSelect">
          <option value="cifar10">CIFAR-10</option>
          <option value="cifar100">CIFAR-100</option>
        </select>
      </label>

      <label>
        Training Duration
        <select id="epochSelect">
          <option value="50">50 Epochs</option>
          <option value="80">80 Epochs</option>
        </select>
      </label>

      <div class="toolbar-description">
        Same dataset and architecture across E1–E4
      </div>

    </section>

    <p class="comparison-status" id="statusMessage">
      Loading experiment data...
    </p>


    <section class="view-toolbar">
      <label for="viewSelect">Chart View</label>

      <select id="viewSelect">
        <option value="four">Four Individual Charts</option>
        <option value="combined">Combined E1–E4 Comparison</option>
      </select>
    </section>

    <section id="combinedPanel" class="combined-panel" hidden>
      <h2>Validation Accuracy Comparison</h2>
      <p>
        All four optimization strategies on the same graph.
        Each line represents one experiment.
      </p>

      <div class="combined-chart-container">
        <canvas id="combinedChart"></canvas>
      </div>
    </section>


    <main class="comparison-grid">

      ${experiments.map(experiment => `
        <article class="experiment-card">

          <div class="experiment-heading">
            <span
              class="experiment-indicator"
              style="background: ${experiment.color}"
            ></span>

            <div>
              <h2>${experiment.id}</h2>
              <p>${experiment.name}</p>
            </div>
          </div>

          <div class="experiment-chart">
            <canvas id="chart-${experiment.id}"></canvas>
          </div>

          <div class="experiment-metrics">
            <div>
              <span>Validation Accuracy</span>
              <strong id="accuracy-${experiment.id}">--</strong>
            </div>

            <div>
              <span>Validation Loss</span>
              <strong id="loss-${experiment.id}">--</strong>
            </div>

            <div>
              <span>Learning Rate</span>
              <strong id="lr-${experiment.id}">--</strong>
            </div>

            <div>
              <span>Batch Size</span>
              <strong id="batch-${experiment.id}">--</strong>
            </div>
          </div>

        </article>
      `).join('')}

    </main>

    <section class="dynamics-section">

      <div class="dynamics-header">
        <div>
          <h2>Optimization Dynamics</h2>
          <p>Learning rate and batch size during training</p>
        </div>
      </div>

      <div class="dynamics-grid">

        <div class="dynamics-card">
          <h3>Learning Rate vs. Epoch</h3>
          <div class="dynamics-chart">
            <canvas id="lrChart"></canvas>
          </div>
        </div>

        <div class="dynamics-card">
          <h3>Batch Size vs. Epoch</h3>
          <div class="dynamics-chart">
            <canvas id="batchChart"></canvas>
          </div>
        </div>

      </div>

      <div class="events-panel">
        <h3>Controller Events</h3>
        <p id="controllerEvents">Select an experiment and press Play to observe changes.</p>
      </div>

    </section>

    <section class="processing-section" aria-labelledby="processingHeading">
      <div class="processing-header">
        <div>
          <h2 id="processingHeading">Processing Time</h2>
          <p>Cumulative recorded training time, averaged across learning-rate strategies</p>
        </div>
      </div>

      <div class="processing-chart">
        <canvas id="processingChart" aria-label="Cumulative processing time comparing fixed and adaptive batch sizing"></canvas>
      </div>
      <p class="processing-summary" id="processingSummary" aria-live="polite"></p>
      <p class="processing-note">This compares recorded training time, not isolated hyperparameter-tuning time. Other run-to-run factors may also affect elapsed time.</p>
    </section>

    <section class="results-section" id="resultsSection" hidden>

      <header class="results-header">
        <div>
          <h2>Experiment Results &amp; Analysis</h2>
          <p>Full-run summary from all recorded epochs, independent of replay position.</p>
        </div>
        <span class="results-tag">Validation-based comparison</span>
      </header>

      <div class="analysis-findings" id="analysisFindings"></div>

      <section class="accuracy-differences" aria-labelledby="accuracyDifferencesHeading">
        <div>
          <h3 id="accuracyDifferencesHeading">Validation Accuracy Differences</h3>
          <p>Percentage-point differences relative to E1. Positive values indicate higher accuracy.</p>
        </div>
        <div class="difference-grid" id="accuracyDifferences"></div>
      </section>

      <div class="results-legend" aria-label="Table highlighting key">
        <span><i class="performance-swatch" aria-hidden="true"></i>Best measured performance</span>
        <span><i class="cost-swatch" aria-hidden="true"></i>Lowest recorded computational cost</span>
        <span>Ties are highlighted together</span>
      </div>

      <div class="results-table-wrap">
        <table class="results-table">
          <caption>Complete-run metrics calculated from the selected CSV files</caption>
          <thead>
            <tr>
              <th scope="col">Metric</th>
              ${experiments.map(experiment => `<th scope="col">${experiment.id}</th>`).join('')}
            </tr>
          </thead>
          <tbody id="resultsTableBody"></tbody>
        </table>
      </div>

      <div class="analysis-notes">
        <p>Accuracy values are stored in the CSVs as percentages (0–100) and are displayed without rescaling. Optimizer updates and elapsed seconds are cumulative in these runs; totals use the final row after checking that each series is present and nondecreasing.</p>
        <p>Highlighted accuracy and validation-loss cells identify the highest or lowest recorded value, with ties retained. Lowest update counts and elapsed time indicate lower recorded computational cost only, not better overall performance. Generalization gap is shown without ranking because a smaller gap can also reflect underfitting. Comparisons remain subject to differences in optimizer updates, training time, and experimental controls. No test accuracy is used.</p>
      </div>

    </section>

    <footer class="controls">

      <button id="playBtn" type="button">▶ Play All</button>
      <button id="resetBtn" type="button">↺ Reset</button>

      <label for="speedSelect">Speed</label>
      <select id="speedSelect">
        <option value="600">0.5×</option>
        <option value="300" selected>1×</option>
        <option value="150">2×</option>
        <option value="60">5×</option>
      </select>

      <input
        type="range"
        id="epochSlider"
        min="0"
        max="50"
        value="0"
        aria-label="Training epoch"
      />

      <span id="epochLabel">Epoch 0 / 50</span>

    </footer>

  </div>
`;

// =============================================
// 3. APPLICATION STATE
// =============================================

const datasetSelect = document.querySelector('#datasetSelect');
const epochSelect = document.querySelector('#epochSelect');
const epochSlider = document.querySelector('#epochSlider');
const epochLabel = document.querySelector('#epochLabel');
const playBtn = document.querySelector('#playBtn');
const resetBtn = document.querySelector('#resetBtn');
const speedSelect = document.querySelector('#speedSelect');
const statusMessage = document.querySelector('#statusMessage');

// Data and chart for each experiment
let experimentData = {};
let charts = {};

let combinedChart = null;
let lrChart = null;
let batchChart = null;
let processingChart = null;

const viewSelect = document.querySelector('#viewSelect');
const comparisonGrid = document.querySelector('.comparison-grid');
const combinedPanel = document.querySelector('#combinedPanel');
const controllerEvents = document.querySelector('#controllerEvents');
const resultsSection = document.querySelector('#resultsSection');
const analysisFindings = document.querySelector('#analysisFindings');
const accuracyDifferences = document.querySelector('#accuracyDifferences');
const resultsTableBody = document.querySelector('#resultsTableBody');

// Shared playback state
let currentEpoch = 0;
let totalEpochs = 0;
let playbackTimer = null;
let playbackSpeed = Number(speedSelect.value);

// Used to ignore outdated loading requests
let loadRequestId = 0;

// =============================================
// 4. EPOCH OPTIONS
// =============================================

function updateEpochOptions() {
  const available = datasetSelect.value === 'cifar100'
    ? [120]
    : [50, 80];

  epochSelect.innerHTML = available.map(count => `
    <option value="${count}">${count} Epochs</option>
  `).join('');
}

// =============================================
// 5. CREATE FOUR CHARTS
// =============================================

function createCharts() {

  // Remove previous Chart.js instances
  Object.values(charts).forEach(chart => chart.destroy());
  charts = {};

  experiments.forEach(experiment => {

    const data = experimentData[experiment.id];

    const canvas = document.querySelector(
      `#chart-${experiment.id}`
    );

    charts[experiment.id] = new Chart(canvas, {
      type: 'line',

      data: {
        labels: data.map(row => row.epoch),

        datasets: [
          {
            label: 'Training Accuracy',
            data: [],
            borderColor: '#34d399',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.15
          },
          {
            label: 'Validation Accuracy',
            data: [],
            borderColor: experiment.color,
            borderWidth: 2.5,
            pointRadius: 0,
            tension: 0.15
          }
        ]
      },

      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,

        interaction: {
          mode: 'index',
          intersect: false
        },

        plugins: {
          legend: {
            labels: {
              color: '#cbd5e1',
              boxWidth: 12
            }
          }
        },

        scales: {
          x: {
            title: {
              display: true,
              text: 'Epoch',
              color: '#94a3b8'
            },
            ticks: {
              color: '#94a3b8',
              maxTicksLimit: 7
            },
            grid: {
              color: '#1e293b'
            }
          },

          y: {
            min: 0,
            max: 100,
            title: {
              display: true,
              text: 'Accuracy (%)',
              color: '#94a3b8'
            },
            ticks: {
              color: '#94a3b8'
            },
            grid: {
              color: '#334155'
            }
          }
        }
      }
    });
  });
}


// =============================================
// COMBINED E1-E4 CHART
// =============================================

function createCombinedChart() {

  if (combinedChart) {
    combinedChart.destroy();
  }

  const canvas = document.querySelector('#combinedChart');

  // All experiments use the same epoch numbers.
  const labels = experimentData.E1.map(row => row.epoch);

  const datasets = experiments.map(experiment => ({
    label: `${experiment.id} — ${experiment.name}`,

    // Start with no visible data.
    data: Array(totalEpochs).fill(null),

    borderColor: experiment.color,
    backgroundColor: experiment.color,

    borderWidth: 2.5,
    pointRadius: 0,
    pointHoverRadius: 5,
    tension: 0.15,
    spanGaps: false
  }));

  combinedChart = new Chart(canvas, {
    type: 'line',

    data: {
      labels,
      datasets
    },

    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,

      interaction: {
        mode: 'index',
        intersect: false
      },

      plugins: {
        legend: {
          position: 'top',
          labels: {
            color: '#f8fafc',
            boxWidth: 14,
            padding: 18
          }
        },

        tooltip: {
          callbacks: {
            label: context => {
              return (
                `${context.dataset.label}: ` +
                `${context.parsed.y.toFixed(2)}%`
              );
            }
          }
        }
      },

      scales: {
        x: {
          title: {
            display: true,
            text: 'Training Epoch',
            color: '#94a3b8'
          },
          ticks: {
            color: '#94a3b8',
            maxTicksLimit: 12
          },
          grid: {
            color: '#1e293b'
          }
        },

        y: {
          min: 0,
          max: 100,

          title: {
            display: true,
            text: 'Validation Accuracy (%)',
            color: '#94a3b8'
          },
          ticks: {
            color: '#94a3b8'
          },
          grid: {
            color: '#334155'
          }
        }
      }
    }
  });
}

function createDynamicsCharts() {
  if (lrChart) lrChart.destroy();
  if (batchChart) batchChart.destroy();

  const labels = experimentData.E1.map(row => row.epoch);
  const createDatasets = label => experiments.map(experiment => ({
    label: `${experiment.id} ${label}`,
    data: Array(totalEpochs).fill(null),
    borderColor: experiment.color,
    backgroundColor: experiment.color,
    borderWidth: 2.5,
    pointRadius: 0,
    pointHoverRadius: 4,
    stepped: true
  }));

  const chartOptions = (title, beginAtZero = false) => ({
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: {
      legend: {
        position: 'top',
        labels: { color: '#cbd5e1', boxWidth: 12 }
      },
      tooltip: { mode: 'index', intersect: false }
    },
    scales: {
      x: {
        title: { display: true, text: 'Epoch', color: '#94a3b8' },
        ticks: { color: '#94a3b8', maxTicksLimit: 8 },
        grid: { color: '#1e293b' }
      },
      y: {
        beginAtZero,
        title: { display: true, text: title, color: '#94a3b8' },
        ticks: { color: '#94a3b8' },
        grid: { color: '#334155' }
      }
    }
  });

  lrChart = new Chart(document.querySelector('#lrChart'), {
    type: 'line',
    data: {
      labels,
      datasets: createDatasets('learning rate')
    },
    options: chartOptions('Learning rate')
  });

  batchChart = new Chart(document.querySelector('#batchChart'), {
    type: 'line',
    data: {
      labels,
      datasets: createDatasets('batch size')
    },
    options: chartOptions('Batch size', true)
  });
}

function createProcessingChart() {
  if (processingChart) processingChart.destroy();

  const labels = experimentData.E1.map(row => row.epoch);
  const averageElapsed = experimentIds => labels.map((_, index) => {
    const values = experimentIds.map(id => experimentData[id][index].elapsedSeconds);
    return values.every(Number.isFinite)
      ? values.reduce((sum, value) => sum + value, 0) / values.length
      : null;
  });

  const fixedBatch = averageElapsed(['E1', 'E3']);
  const adaptiveBatch = averageElapsed(['E2', 'E4']);
  const fixedFinal = fixedBatch.at(-1);
  const adaptiveFinal = adaptiveBatch.at(-1);
  const timeSavedPercent = Number.isFinite(fixedFinal) && fixedFinal > 0 && Number.isFinite(adaptiveFinal)
    ? ((fixedFinal - adaptiveFinal) / fixedFinal) * 100
    : null;

  document.querySelector('#processingSummary').textContent = Number.isFinite(timeSavedPercent)
    ? `At epoch ${labels.at(-1)}, adaptive batch sizing averaged ${(adaptiveFinal / 60).toFixed(2)} min vs ${(fixedFinal / 60).toFixed(2)} min with fixed batch sizing (${timeSavedPercent.toFixed(1)}% less recorded time).`
    : 'Recorded timing data is unavailable for this comparison.';

  processingChart = new Chart(document.querySelector('#processingChart'), {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Fixed batch (E1 + E3 average)',
          data: fixedBatch,
          borderColor: '#60a5fa',
          backgroundColor: '#60a5fa',
          borderWidth: 2.5,
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0.15
        },
        {
          label: 'Adaptive batch (E2 + E4 average)',
          data: adaptiveBatch,
          borderColor: '#34d399',
          backgroundColor: '#34d399',
          borderWidth: 2.5,
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0.15
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          position: 'top',
          labels: { color: '#cbd5e1', boxWidth: 12 }
        },
        tooltip: {
          callbacks: {
            label: context => `${context.dataset.label}: ${context.parsed.y.toFixed(1)} s`
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: 'Epoch', color: '#94a3b8' },
          ticks: { color: '#94a3b8', maxTicksLimit: 10 },
          grid: { color: '#1e293b' }
        },
        y: {
          beginAtZero: true,
          title: { display: true, text: 'Cumulative elapsed time (seconds)', color: '#94a3b8' },
          ticks: { color: '#94a3b8' },
          grid: { color: '#334155' }
        }
      }
    }
  });
}

function updateDynamicsCharts() {
  if (!lrChart || !batchChart) return;

  experiments.forEach((experiment, index) => {
    const data = experimentData[experiment.id];

    lrChart.data.datasets[index].data = data.map((row, epochIndex) =>
      epochIndex < currentEpoch ? row.learningRate : null
    );
    batchChart.data.datasets[index].data = data.map((row, epochIndex) =>
      epochIndex < currentEpoch ? row.batchSize : null
    );
  });

  lrChart.update('none');
  batchChart.update('none');

  updateControllerEvents();
}

function updateControllerEvents() {
  if (currentEpoch === 0) {
    controllerEvents.textContent = 'No controller events recorded yet.';
    return;
  }

  const events = [];

  for (let epochIndex = 1; epochIndex < currentEpoch; epochIndex++) {
    experiments.forEach(experiment => {
      const data = experimentData[experiment.id];
      const previous = data[epochIndex - 1];
      const current = data[epochIndex];

      if (current.learningRate !== previous.learningRate) {
        const reason = current.lrChangeReason && current.lrChangeReason !== 'keep'
          ? ` (${current.lrChangeReason})`
          : '';
        events.push(
          `Epoch ${current.epoch} · ${experiment.id}: learning rate ${previous.learningRate} -> ${current.learningRate}${reason}`
        );
      }

      if (current.batchSize !== previous.batchSize) {
        const reason = current.batchChangeReason
          ? ` (${current.batchChangeReason})`
          : '';
        events.push(
          `Epoch ${current.epoch} · ${experiment.id}: batch size ${previous.batchSize} -> ${current.batchSize}${reason}`
        );
      }
    });
  }

  controllerEvents.textContent = events.length > 0
    ? `Through epoch ${currentEpoch}: ${events.join('; ')}.`
    : `No learning-rate or batch-size changes recorded through epoch ${currentEpoch}.`;
}

function formatNumber(value, digits = 2) {
  return Number.isFinite(value) ? value.toFixed(digits) : 'N/A';
}

function formatPercent(value) {
  return Number.isFinite(value) ? `${value.toFixed(2)}%` : 'N/A';
}

function formatTrainingTime(seconds) {
  return Number.isFinite(seconds)
    ? `${seconds.toFixed(2)} s (${(seconds / 60).toFixed(2)} min)`
    : 'N/A';
}

function formatPercentagePointDifference(value) {
  if (!Number.isFinite(value)) return 'N/A';

  const sign = value > 0 ? '+' : '';
  return `${sign}${value.toFixed(2)} pp`;
}

function renderExperimentAnalysis() {
  const analysis = analyzeExperiments(experimentData);
  const { metrics, winners } = analysis;

  const bestPeakFinding = winners.bestValidationAccuracy.length > 0
    ? winners.bestValidationAccuracy.map(id => {
      const epochs = metrics[id].bestValidationEpochs;
      const epochText = epochs.length > 0 ? ` at epoch${epochs.length > 1 ? 's' : ''} ${epochs.join(', ')}` : '';
      return `${id} · ${formatPercent(metrics[id].bestValidationAccuracy)}${epochText}`;
    }).join('; ')
    : 'N/A';

  const findingCards = [
    {
      label: 'Best recorded validation accuracy',
      value: bestPeakFinding
    },
    {
      label: 'Best final validation accuracy',
      value: winners.finalValidationAccuracy.length > 0
        ? `${winners.finalValidationAccuracy.join(', ')} · ${formatPercent(metrics[winners.finalValidationAccuracy[0]].finalValidationAccuracy)}`
        : 'N/A'
    },
    {
      label: 'Lowest final validation loss',
      value: winners.finalValidationLoss.length > 0
        ? `${winners.finalValidationLoss.join(', ')} · ${formatNumber(metrics[winners.finalValidationLoss[0]].finalValidationLoss, 4)}`
        : 'N/A'
    },
    {
      label: 'Shortest recorded training time',
      value: winners.recordedTrainingTimeSeconds.length > 0
        ? `${winners.recordedTrainingTimeSeconds.join(', ')} · ${formatTrainingTime(metrics[winners.recordedTrainingTimeSeconds[0]].recordedTrainingTimeSeconds)}`
        : 'N/A'
    }
  ];

  analysisFindings.innerHTML = findingCards.map(finding => `
    <article class="analysis-finding">
      <h3>${finding.label}</h3>
      <p>${finding.value}</p>
    </article>
  `).join('');

  const differenceGroups = [
    { key: 'bestValidationAccuracy', label: 'Best validation accuracy vs E1' },
    { key: 'finalValidationAccuracy', label: 'Final validation accuracy vs E1' }
  ];

  accuracyDifferences.innerHTML = differenceGroups.map(group => `
    <div class="difference-group">
      <h4>${group.label}</h4>
      <ul>
        ${experiments.map(experiment => `
          <li>
            <span>${experiment.id}</span>
            <strong>${formatPercentagePointDifference(analysis.accuracyDifferences[group.key][experiment.id])}</strong>
          </li>
        `).join('')}
      </ul>
    </div>
  `).join('');

  const metricRows = [
    { label: 'Best validation accuracy', key: 'bestValidationAccuracy', format: formatPercent, winner: 'bestValidationAccuracy' },
    {
      label: 'Epoch(s) at best validation accuracy',
      key: 'bestValidationEpochs',
      format: epochs => epochs.length > 0 ? epochs.join(', ') : 'N/A',
      winner: 'bestValidationAccuracy'
    },
    { label: 'Final validation accuracy', key: 'finalValidationAccuracy', format: formatPercent, winner: 'finalValidationAccuracy' },
    { label: 'Final training accuracy', key: 'finalTrainingAccuracy', format: formatPercent, winner: 'finalTrainingAccuracy' },
    { label: 'Final validation loss', key: 'finalValidationLoss', format: value => formatNumber(value, 4), winner: 'finalValidationLoss' },
    { label: 'Total optimizer updates', key: 'totalOptimizerUpdates', format: value => Number.isFinite(value) ? Math.round(value).toLocaleString() : 'N/A', winner: 'totalOptimizerUpdates', cost: true },
    { label: 'Total recorded training time', key: 'recordedTrainingTimeSeconds', format: formatTrainingTime, winner: 'recordedTrainingTimeSeconds', cost: true },
    { label: 'Final-epoch generalization gap', key: 'finalGeneralizationGap', format: value => Number.isFinite(value) ? `${formatNumber(value)} pp` : 'N/A' }
  ];

  resultsTableBody.innerHTML = metricRows.map(row => `
    <tr>
      <th scope="row">${row.label}</th>
      ${experiments.map(experiment => {
        const isWinner = row.winner && winners[row.winner].includes(experiment.id);
        const winnerClass = isWinner
          ? row.cost ? 'cost-winner' : 'metric-winner'
          : '';
        return `<td class="${winnerClass}">${row.format(metrics[experiment.id][row.key])}</td>`;
      }).join('')}
    </tr>
  `).join('');
}


// =============================================
// 6. DISPLAY A SPECIFIC EPOCH
// =============================================

function renderEpoch(epoch) {

  if (totalEpochs === 0) return;

  currentEpoch = Math.max(
    0,
    Math.min(Math.floor(epoch), totalEpochs)
  );

  experiments.forEach(experiment => {

    const id = experiment.id;
    const data = experimentData[id];
    const chart = charts[id];

    // Reveal only the epochs reached so far
    const visibleData = data.slice(0, currentEpoch);

    chart.data.datasets[0].data =
      visibleData.map(row => row.trainAccuracy);

    chart.data.datasets[1].data =
      visibleData.map(row => row.valAccuracy);

    chart.update('none');

    // Update the live metrics for this experiment
    const row = visibleData.at(-1);

    document.querySelector(`#accuracy-${id}`).textContent =
      row ? `${row.valAccuracy.toFixed(2)}%` : '--';

    document.querySelector(`#loss-${id}`).textContent =
      row ? row.valLoss.toFixed(4) : '--';

    document.querySelector(`#lr-${id}`).textContent =
      row ? row.learningRate.toPrecision(4) : '--';

    document.querySelector(`#batch-${id}`).textContent =
      row ? row.batchSize : '--';
  });


    // Update combined comparison chart
    if (combinedChart) {

      experiments.forEach((experiment, index) => {

        const data = experimentData[experiment.id];

        // Keep the full x-axis, reveal epochs progressively.
        combinedChart.data.datasets[index].data =
          data.map((row, i) =>
            i < currentEpoch ? row.valAccuracy : null
          );

      });

      combinedChart.update('none');
    }

  updateDynamicsCharts();


  epochSlider.value = currentEpoch;

  epochLabel.textContent =
    `Epoch ${currentEpoch} / ${totalEpochs}`;
}

// =============================================
// 7. PLAYBACK CONTROLS
// =============================================

function stopPlayback() {
  if (playbackTimer !== null) {
    clearInterval(playbackTimer);
    playbackTimer = null;
  }

  playBtn.textContent = '▶ Play All';
}

function togglePlayback() {

  if (totalEpochs === 0) return;

  // If currently playing, pause
  if (playbackTimer !== null) {
    stopPlayback();
    return;
  }

  // Start again if at the end
  if (currentEpoch >= totalEpochs) {
    renderEpoch(0);
  }

  playBtn.textContent = '⏸ Pause All';

  playbackTimer = setInterval(() => {

    renderEpoch(currentEpoch + 1);

    if (currentEpoch >= totalEpochs) {
      stopPlayback();
    }

  }, playbackSpeed);
}

function resetPlayback() {
  stopPlayback();

  if (totalEpochs > 0) {
    renderEpoch(0);
  }
}

// =============================================
// 8. LOAD ALL FOUR EXPERIMENTS
// =============================================

async function loadComparison() {

  stopPlayback();

  const requestId = ++loadRequestId;

  // Disable playback until data is loaded
  totalEpochs = 0;
  playBtn.disabled = true;
  resetBtn.disabled = true;
  epochSlider.disabled = true;

  statusMessage.textContent = 'Loading four experiments...';
  resultsSection.hidden = true;

  const dataset = datasetSelect.value;
  const epochs = Number(epochSelect.value);

  try {

    // Load E1, E2, E3 and E4 concurrently
    const results = await Promise.all(
      experiments.map(experiment =>
        loadExperiment(dataset, epochs, experiment.id)
      )
    );

    // Ignore an outdated request
    if (requestId !== loadRequestId) return;

    // Associate each result with its experiment ID
    const loadedData = {};

    experiments.forEach((experiment, index) => {
      loadedData[experiment.id] = results[index];
    });

    // Verify all four runs contain the expected epochs
    for (const experiment of experiments) {

      const data = loadedData[experiment.id];

      if (
        data.length !== epochs ||
        data.some((row, index) => row.epoch !== index + 1)
      ) {
        throw new Error(
          `${experiment.id} does not contain the expected ${epochs} epochs`
        );
      }
    }

    experimentData = loadedData;
    totalEpochs = epochs;

    renderExperimentAnalysis();
    resultsSection.hidden = false;

    createCharts();
    createCombinedChart();
    createDynamicsCharts();
    createProcessingChart();

    epochSlider.max = totalEpochs;

    renderEpoch(0);
    updateChartView();

    playBtn.disabled = false;
    resetBtn.disabled = false;
    epochSlider.disabled = false;

    statusMessage.textContent =
      `${dataset.toUpperCase()} · ${epochs} epochs · All 4 experiments ready`;

  } catch (error) {

    if (requestId !== loadRequestId) return;

    console.error('Comparison loading failed:', error);

    statusMessage.textContent =
      `Could not load experiment data: ${error.message}`;
  }
}

// =============================================
// 9. EVENT LISTENERS
// =============================================

playBtn.addEventListener('click', togglePlayback);

resetBtn.addEventListener('click', resetPlayback);

epochSlider.addEventListener('input', () => {
  stopPlayback();
  renderEpoch(Number(epochSlider.value));
});

speedSelect.addEventListener('change', () => {

  playbackSpeed = Number(speedSelect.value);

  // Restart the timer if already playing
  if (playbackTimer !== null) {
    stopPlayback();
    togglePlayback();
  }
});

datasetSelect.addEventListener('change', () => {
  updateEpochOptions();
  loadComparison();
});

epochSelect.addEventListener('change', loadComparison);



// =============================================
// SWITCH CHART VIEWS
// =============================================

function updateChartView() {

  const isCombined = viewSelect.value === 'combined';

  // Show the selected panel, hide the other.
  comparisonGrid.hidden = isCombined;
  combinedPanel.hidden = !isCombined;

  // Resize charts after making their containers visible.
  if (isCombined && combinedChart) {
    combinedChart.resize();
    combinedChart.update('none');
  }

  if (!isCombined) {
    Object.values(charts).forEach(chart => {
      chart.resize();
      chart.update('none');
    });
  }
}

viewSelect.addEventListener('change', updateChartView);


// =============================================
// 10. START APPLICATION
// =============================================

updateEpochOptions();
loadComparison();
