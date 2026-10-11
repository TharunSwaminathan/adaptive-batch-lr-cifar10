import Chart from 'chart.js/auto';
import {
  calculateDatasetSummary,
  calculateImprovement,
  getEfficiencyPoints
} from '../utils/crossDatasetAnalysis.js';

const ACCURACY_METRICS = {
  bestValidationAccuracy: 'Best Validation Accuracy',
  finalValidationAccuracy: 'Final Validation Accuracy'
};

const EFFICIENCY_AXES = {
  epoch: { key: 'epoch', label: 'Epoch number', unit: 'epoch' },
  optimizerUpdates: { key: 'optimizerUpdates', label: 'Cumulative optimizer updates', unit: 'updates' },
  elapsedSeconds: { key: 'elapsedSeconds', label: 'Cumulative elapsed training time', unit: 'seconds' }
};

function formatAccuracy(value) {
  return Number.isFinite(value) ? `${value.toFixed(2)}%` : 'N/A';
}

function formatDifference(value) {
  if (!Number.isFinite(value)) return 'N/A';
  return `${value > 0 ? '+' : ''}${value.toFixed(2)} pp`;
}

function formatUpdates(value) {
  return Number.isFinite(value) ? Math.round(value).toLocaleString() : 'N/A';
}

function formatTime(seconds) {
  return Number.isFinite(seconds)
    ? `${seconds.toFixed(2)} s (${(seconds / 60).toFixed(2)} min)`
    : 'N/A';
}

function formatWinner(ids, metrics, metric, formatter) {
  if (ids.length === 0) return 'N/A';
  const value = metrics[ids[0]]?.[metric];
  return `${ids.join(', ')} · ${formatter(value)}`;
}

export function mountCrossDatasetView({ container, experiments, loadExperiment }) {
  container.innerHTML = `
    <div class="cross-dataset-view">
      <header class="cross-dataset-heading">
        <div>
          <h2>Cross-Dataset Analysis</h2>
          <p>Compare recorded validation behavior and efficiency within each dataset.</p>
        </div>
        <span class="cross-dataset-status" id="crossDatasetStatus" role="status">Loading recorded runs...</span>
      </header>

      <div class="cross-dataset-controls">
        <label>
          CIFAR-10 duration
          <select id="crossCifar10Epochs">
            <option value="80" selected>80 epochs</option>
            <option value="50">50 epochs</option>
          </select>
        </label>
        <label>
          Accuracy metric
          <select id="crossAccuracyMetric">
            <option value="bestValidationAccuracy" selected>Best validation accuracy</option>
            <option value="finalValidationAccuracy">Final validation accuracy</option>
          </select>
        </label>
      </div>

      <section class="cross-analysis-section" aria-labelledby="crossAccuracyHeading">
        <div class="cross-section-heading">
          <div>
            <h3 id="crossAccuracyHeading">Validation Accuracy by Epoch</h3>
            <p>Separate epoch axes retain each run's recorded training budget.</p>
          </div>
        </div>
        <div class="cross-accuracy-grid">
          <article class="cross-chart-panel">
            <header class="cross-chart-heading">
              <h4>CIFAR-10</h4>
              <span id="crossCifar10DurationLabel">80 epochs</span>
            </header>
            <div class="cross-chart-container cross-accuracy-chart">
              <canvas id="crossCifar10Chart" aria-label="CIFAR-10 validation accuracy by epoch"></canvas>
            </div>
          </article>
          <article class="cross-chart-panel">
            <header class="cross-chart-heading">
              <h4>CIFAR-100</h4>
              <span>120 epochs</span>
            </header>
            <div class="cross-chart-container cross-accuracy-chart">
              <canvas id="crossCifar100Chart" aria-label="CIFAR-100 validation accuracy by epoch"></canvas>
            </div>
          </article>
        </div>
      </section>

      <section class="cross-analysis-section" aria-labelledby="crossImprovementHeading">
        <div class="cross-section-heading">
          <div>
            <h3 id="crossImprovementHeading">Improvement Relative to E1</h3>
            <p>Strategy accuracy minus E1 accuracy, in percentage points (pp).</p>
          </div>
        </div>
        <div class="cross-chart-container cross-improvement-chart">
          <canvas id="crossImprovementChart" aria-label="Grouped bar chart of validation accuracy percentage-point differences from E1"></canvas>
        </div>
        <p class="cross-chart-note">Bars above zero indicate higher recorded validation accuracy than E1; bars below zero indicate lower accuracy. This is a descriptive comparison, not a significance test.</p>
      </section>

      <section class="cross-analysis-section" aria-labelledby="crossEfficiencyHeading">
        <div class="cross-section-heading cross-efficiency-heading">
          <div>
            <h3 id="crossEfficiencyHeading">Validation Accuracy vs. Training Progress</h3>
            <p>Points are recorded observations only; highlighted endpoints mark the last valid epoch.</p>
          </div>
          <div class="cross-efficiency-controls">
            <label>
              Dataset
              <select id="crossEfficiencyDataset">
                <option value="cifar10" selected>CIFAR-10</option>
                <option value="cifar100">CIFAR-100</option>
              </select>
            </label>
            <label>
              Duration
              <select id="crossEfficiencyEpochs">
                <option value="80" selected>80 epochs</option>
                <option value="50">50 epochs</option>
              </select>
            </label>
            <label>
              Horizontal axis
              <select id="crossEfficiencyAxis">
                <option value="epoch" selected>Epoch number</option>
                <option value="optimizerUpdates">Optimizer updates</option>
                <option value="elapsedSeconds">Elapsed time</option>
              </select>
            </label>
          </div>
        </div>
        <div class="cross-chart-container cross-efficiency-chart">
          <canvas id="crossEfficiencyChart" aria-label="Recorded validation accuracy plotted against the selected training progress measure"></canvas>
        </div>
        <p class="cross-chart-note" id="crossEfficiencyNote">Optimizer updates and elapsed time are cumulative values from the CSV; they are not summed again.</p>
      </section>

      <section class="cross-analysis-section" aria-labelledby="crossSummaryHeading">
        <div class="cross-section-heading">
          <div>
            <h3 id="crossSummaryHeading">Cross-Dataset Results Summary</h3>
            <p>Accuracy rows rank validation performance; efficiency rows rank the lowest recorded totals. Ties are retained.</p>
          </div>
        </div>
        <div class="cross-summary-table-wrap">
          <table class="cross-summary-table">
            <caption>Complete-run metrics from all recorded epochs</caption>
            <thead>
              <tr>
                <th scope="col">Metric</th>
                <th scope="col" id="crossCifar10SummaryHeading">CIFAR-10 · 80 epochs</th>
                <th scope="col">CIFAR-100 · 120 epochs</th>
              </tr>
            </thead>
            <tbody id="crossSummaryBody"></tbody>
          </table>
        </div>
      </section>

      <aside class="cross-scientific-notes" aria-label="Interpretation and limitations">
        <h3>Interpretation and Limitations</h3>
        <p>Compare each adaptive strategy with E1 within the same dataset. CIFAR-10 and CIFAR-100 differ in class count and task difficulty, and the 80-epoch CIFAR-10 and 120-epoch CIFAR-100 runs use different training budgets; absolute accuracy across datasets is not an estimate of the hyperparameter effect.</p>
        <p>These are recorded single-run observations, not statistical-significance tests. Fewer optimizer updates do not necessarily mean less computation because batch sizes differ. Elapsed-time results are specific to these recorded runs and hardware. No test-set performance is used, and controller events alone do not establish causality.</p>
      </aside>
    </div>
  `;

  const ids = experiments.map(experiment => experiment.id);
  const runCache = new Map();
  const accuracyMetricSelect = container.querySelector('#crossAccuracyMetric');
  const cifar10EpochSelect = container.querySelector('#crossCifar10Epochs');
  const efficiencyDatasetSelect = container.querySelector('#crossEfficiencyDataset');
  const efficiencyEpochSelect = container.querySelector('#crossEfficiencyEpochs');
  const efficiencyAxisSelect = container.querySelector('#crossEfficiencyAxis');
  const status = container.querySelector('#crossDatasetStatus');
  const charts = {};
  let requestId = 0;

  function getRuns(dataset, epochs) {
    const key = `${dataset}:${epochs}`;
    if (!runCache.has(key)) {
      const promise = Promise.all(experiments.map(experiment =>
        loadExperiment(dataset, epochs, experiment.id)
      )).then(results => Object.fromEntries(experiments.map((experiment, index) => [
        experiment.id,
        results[index]
      ]))).catch(error => {
        runCache.delete(key);
        throw error;
      });
      runCache.set(key, promise);
    }
    return runCache.get(key);
  }

  function createOrUpdateChart(key, canvasId, config) {
    if (!charts[key]) {
      charts[key] = new Chart(container.querySelector(`#${canvasId}`), config);
      return charts[key];
    }

    charts[key].data = config.data;
    charts[key].options = config.options;
    charts[key].update('none');
    return charts[key];
  }

  function accuracyChartConfig(runs, epochs) {
    return {
      type: 'line',
      data: {
        datasets: experiments.map(experiment => ({
          label: `${experiment.id} · ${experiment.name}`,
          data: (runs[experiment.id] ?? [])
            .filter(row => Number.isFinite(row.epoch) && Number.isFinite(row.valAccuracy))
            .map(row => ({ x: row.epoch, y: row.valAccuracy })),
          borderColor: experiment.color,
          backgroundColor: experiment.color,
          borderWidth: 2.25,
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0.12,
          spanGaps: false
        }))
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: 'nearest', intersect: false },
        plugins: {
          legend: {
            position: 'top',
            labels: { color: '#cbd5e1', boxWidth: 11, padding: 12 }
          },
          tooltip: {
            callbacks: {
              title: items => items.length ? `Epoch ${items[0].parsed.x}` : '',
              label: context => `${context.dataset.label}: ${context.parsed.y.toFixed(2)}% validation accuracy`
            }
          }
        },
        scales: {
          x: {
            type: 'linear',
            min: 1,
            max: epochs,
            title: { display: true, text: 'Recorded epoch', color: '#94a3b8' },
            ticks: { color: '#94a3b8', maxTicksLimit: 9, precision: 0 },
            grid: { color: '#1e293b' }
          },
          y: {
            min: 0,
            max: 100,
            title: { display: true, text: 'Validation accuracy (%)', color: '#94a3b8' },
            ticks: { color: '#94a3b8' },
            grid: { color: '#334155' }
          }
        }
      }
    };
  }

  function renderAccuracyCharts(cifar10Runs, cifar10Epochs, cifar100Runs) {
    container.querySelector('#crossCifar10DurationLabel').textContent = `${cifar10Epochs} epochs`;
    container.querySelector('#crossCifar10SummaryHeading').textContent = `CIFAR-10 · ${cifar10Epochs} epochs`;

    createOrUpdateChart('cifar10Accuracy', 'crossCifar10Chart', accuracyChartConfig(cifar10Runs, cifar10Epochs));
    createOrUpdateChart('cifar100Accuracy', 'crossCifar100Chart', accuracyChartConfig(cifar100Runs, 120));
  }

  function improvementChartConfig(cifar10Runs, cifar10Epochs, cifar100Runs) {
    const metric = accuracyMetricSelect.value;
    const cifar10 = calculateImprovement(cifar10Runs, ids, metric);
    const cifar100 = calculateImprovement(cifar100Runs, ids, metric);
    const details = { cifar10, cifar100 };
    const labels = [`CIFAR-10 · ${cifar10Epochs} epochs`, 'CIFAR-100 · 120 epochs'];
    const adaptiveExperiments = experiments.filter(experiment => experiment.id !== 'E1');

    return {
      details,
      config: {
        type: 'bar',
        data: {
          labels,
          datasets: adaptiveExperiments.map(experiment => ({
            label: `${experiment.id} · ${experiment.name}`,
            data: [
              cifar10.strategies[experiment.id]?.difference ?? null,
              cifar100.strategies[experiment.id]?.difference ?? null
            ],
            backgroundColor: [cifar10, cifar100].map(dataset => {
              const difference = dataset.strategies[experiment.id]?.difference;
              if (!Number.isFinite(difference)) return '#64748b66';
              return difference < 0 ? '#fb718588' : `${experiment.color}bb`;
            }),
            borderColor: [cifar10, cifar100].map(dataset => {
              const difference = dataset.strategies[experiment.id]?.difference;
              if (!Number.isFinite(difference)) return '#64748b';
              return difference < 0 ? '#fb7185' : experiment.color;
            }),
            borderWidth: 1.5
          }))
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          interaction: { mode: 'index', intersect: false },
          plugins: {
            legend: {
              position: 'top',
              labels: { color: '#cbd5e1', boxWidth: 12, padding: 16 }
            },
            tooltip: {
              callbacks: {
                label: context => {
                  const datasetKey = context.dataIndex === 0 ? 'cifar10' : 'cifar100';
                  const strategy = details[datasetKey].strategies[adaptiveExperiments[context.datasetIndex].id];
                  return [
                    `${context.dataset.label}: ${formatDifference(strategy?.difference)}`,
                    `E1 baseline: ${formatAccuracy(strategy?.baselineAccuracy)}`,
                    `Strategy accuracy: ${formatAccuracy(strategy?.accuracy)}`
                  ];
                }
              }
            }
          },
          scales: {
            x: {
              title: { display: true, text: metric === 'bestValidationAccuracy' ? 'Dataset and training duration · best validation accuracy' : 'Dataset and training duration · final validation accuracy', color: '#94a3b8' },
              ticks: { color: '#cbd5e1', maxRotation: 0, minRotation: 0 },
              grid: { display: false }
            },
            y: {
              beginAtZero: true,
              title: { display: true, text: 'Difference from E1 (percentage points)', color: '#94a3b8' },
              ticks: {
                color: '#94a3b8',
                callback: value => `${value > 0 ? '+' : ''}${value} pp`
              },
              grid: {
                color: context => context.tick.value === 0 ? '#f8fafc' : '#334155',
                lineWidth: context => context.tick.value === 0 ? 2 : 1
              }
            }
          }
        }
      }
    };
  }

  function renderImprovementChart(cifar10Runs, cifar10Epochs, cifar100Runs) {
    const { config } = improvementChartConfig(cifar10Runs, cifar10Epochs, cifar100Runs);
    createOrUpdateChart('improvement', 'crossImprovementChart', config);
  }

  function efficiencyChartConfig(runs, dataset, epochs, axisKey) {
    const axis = EFFICIENCY_AXES[axisKey];
    const datasets = experiments.map(experiment => {
      const rows = runs[experiment.id] ?? [];
      const points = getEfficiencyPoints(rows, axis.key);
      const finalEpoch = rows.at(-1)?.epoch;
      return {
        label: `${experiment.id} · ${experiment.name}`,
        data: points,
        showLine: false,
        pointRadius: context => context.raw?.epoch === finalEpoch ? 5 : 2.5,
        pointHoverRadius: 5,
        pointBackgroundColor: experiment.color,
        pointBorderColor: context => context.raw?.epoch === finalEpoch ? '#f8fafc' : experiment.color,
        pointBorderWidth: context => context.raw?.epoch === finalEpoch ? 1.5 : 0.5,
        hitRadius: 5
      };
    });

    return {
      type: 'scatter',
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        parsing: false,
        interaction: { mode: 'nearest', intersect: false },
        plugins: {
          legend: {
            position: 'top',
            labels: { color: '#cbd5e1', boxWidth: 11, padding: 12 }
          },
          tooltip: {
            callbacks: {
              title: items => items.length ? `Epoch ${items[0].raw.epoch}` : '',
              label: context => `${context.dataset.label}: ${context.raw.y.toFixed(2)}% at ${context.raw.x.toLocaleString()} ${axis.unit}`
            }
          }
        },
        scales: {
          x: {
            type: 'linear',
            title: { display: true, text: axis.label, color: '#94a3b8' },
            ticks: {
              color: '#94a3b8',
              callback: value => axisKey === 'elapsedSeconds'
                ? Number(value).toLocaleString(undefined, { maximumFractionDigits: 0 })
                : Number(value).toLocaleString()
            },
            grid: { color: '#1e293b' }
          },
          y: {
            min: 0,
            max: 100,
            title: { display: true, text: 'Validation accuracy (%)', color: '#94a3b8' },
            ticks: { color: '#94a3b8' },
            grid: { color: '#334155' }
          }
        }
      },
      label: `${dataset.toUpperCase()} · ${epochs} epochs`
    };
  }

  function renderEfficiencyChart(runs, dataset, epochs) {
    const axisKey = efficiencyAxisSelect.value;
    const { type, data, options, label } = efficiencyChartConfig(runs, dataset, epochs, axisKey);
    const config = { type, data, options };
    createOrUpdateChart('efficiency', 'crossEfficiencyChart', config);
    container.querySelector('#crossEfficiencyHeading').setAttribute(
      'aria-label',
      `Validation accuracy vs. training progress for ${label}`
    );
    container.querySelector('#crossEfficiencyNote').textContent = axisKey === 'optimizerUpdates'
      ? `${label}. Optimizer-update values are cumulative; they are plotted as recorded and not summed again. Fewer updates do not necessarily mean less computation because batch sizes differ.`
      : axisKey === 'elapsedSeconds'
        ? `${label}. Elapsed-time values are cumulative; they are plotted as recorded and not summed again. Timing reflects these runs and hardware.`
        : `${label}. Epoch numbers are the recorded observations; no missing accuracy observations are interpolated.`;
  }

  function renderSummary(cifar10Runs, cifar10Epochs, cifar100Runs) {
    const summaries = {
      cifar10: calculateDatasetSummary(cifar10Runs, ids),
      cifar100: calculateDatasetSummary(cifar100Runs, ids)
    };
    const columns = [
      { key: 'cifar10', label: `CIFAR-10 · ${cifar10Epochs} epochs` },
      { key: 'cifar100', label: 'CIFAR-100 · 120 epochs' }
    ];
    const rows = [
      {
        label: 'Best strategy by peak validation accuracy',
        group: 'accuracy',
        value: summary => formatWinner(summary.peakWinners, summary.metrics, 'bestValidationAccuracy', formatAccuracy)
      },
      ...['E2', 'E3', 'E4'].map(id => ({
        label: `${id} gain over E1 · best validation accuracy`,
        group: 'accuracy',
        value: summary => formatDifference(summary.peakImprovements.strategies[id]?.difference)
      })),
      {
        label: 'E4 vs. best single-adaptation strategy · peak accuracy',
        group: 'accuracy',
        value: summary => formatDifference(summary.e4VsBestSingle)
      },
      {
        label: 'Fewest recorded optimizer updates',
        group: 'efficiency',
        value: summary => formatWinner(summary.efficiencyWinners.optimizerUpdates, summary.metrics, 'totalOptimizerUpdates', formatUpdates)
      },
      {
        label: 'Shortest recorded training time',
        group: 'efficiency',
        value: summary => formatWinner(summary.efficiencyWinners.elapsedSeconds, summary.metrics, 'recordedTrainingTimeSeconds', formatTime)
      }
    ];

    container.querySelector('#crossSummaryBody').innerHTML = rows.map(row => `
      <tr class="cross-summary-${row.group}">
        <th scope="row">${row.label}</th>
        ${columns.map(column => `<td>${row.value(summaries[column.key])}</td>`).join('')}
      </tr>
    `).join('');
    container.querySelector('#crossCifar10SummaryHeading').textContent = columns[0].label;
  }

  async function refresh() {
    const currentRequest = ++requestId;
    const cifar10Epochs = Number(cifar10EpochSelect.value);
    const efficiencyDataset = efficiencyDatasetSelect.value;
    const efficiencyEpochs = Number(efficiencyEpochSelect.value);

    status.textContent = 'Loading recorded runs...';
    status.dataset.state = 'loading';

    try {
      const [cifar10Runs, cifar100Runs, efficiencyRuns] = await Promise.all([
        getRuns('cifar10', cifar10Epochs),
        getRuns('cifar100', 120),
        getRuns(efficiencyDataset, efficiencyEpochs)
      ]);

      if (currentRequest !== requestId) return;

      renderAccuracyCharts(cifar10Runs, cifar10Epochs, cifar100Runs);
      renderImprovementChart(cifar10Runs, cifar10Epochs, cifar100Runs);
      renderEfficiencyChart(efficiencyRuns, efficiencyDataset, efficiencyEpochs);
      renderSummary(cifar10Runs, cifar10Epochs, cifar100Runs);
      status.textContent = 'Recorded runs ready';
      status.dataset.state = 'ready';
    } catch (error) {
      if (currentRequest !== requestId) return;
      status.textContent = `Could not load cross-dataset data: ${error.message}`;
      status.dataset.state = 'error';
    }
  }

  function updateEfficiencyEpochOptions() {
    const choices = efficiencyDatasetSelect.value === 'cifar100' ? [120] : [80, 50];
    const current = Number(efficiencyEpochSelect.value);
    const selected = choices.includes(current) ? current : choices[0];
    efficiencyEpochSelect.innerHTML = choices.map(epochs =>
      `<option value="${epochs}"${epochs === selected ? ' selected' : ''}>${epochs} epochs</option>`
    ).join('');
  }

  cifar10EpochSelect.addEventListener('change', refresh);
  accuracyMetricSelect.addEventListener('change', refresh);
  efficiencyDatasetSelect.addEventListener('change', () => {
    updateEfficiencyEpochOptions();
    refresh();
  });
  efficiencyEpochSelect.addEventListener('change', refresh);
  efficiencyAxisSelect.addEventListener('change', refresh);

  updateEfficiencyEpochOptions();
  refresh();

  return {
    refreshLayout() {
      for (const chart of Object.values(charts)) {
        chart.resize();
        chart.update('none');
      }
    },
    destroy() {
      for (const chart of Object.values(charts)) chart.destroy();
      runCache.clear();
    }
  };
}
