import { calculateExperimentMetrics } from './experimentAnalysis.js';

const ACCURACY_METRICS = new Set([
  'bestValidationAccuracy',
  'finalValidationAccuracy'
]);

const EFFICIENCY_METRICS = {
  optimizerUpdates: 'totalOptimizerUpdates',
  elapsedSeconds: 'recordedTrainingTimeSeconds'
};

function finiteValue(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function findWinners(metricsByExperiment, metric, direction) {
  const values = Object.entries(metricsByExperiment)
    .map(([id, metrics]) => ({ id, value: finiteValue(metrics?.[metric]) }))
    .filter(result => result.value !== null);

  if (values.length === 0) return [];

  const bestValue = direction === 'min'
    ? Math.min(...values.map(result => result.value))
    : Math.max(...values.map(result => result.value));

  return values
    .filter(result => result.value === bestValue)
    .map(result => result.id);
}

export function calculateImprovement(runsByExperiment, experimentIds, metric) {
  if (!ACCURACY_METRICS.has(metric)) {
    throw new Error(`Unsupported accuracy metric: ${metric}`);
  }

  const metrics = Object.fromEntries(experimentIds.map(id => [
    id,
    calculateExperimentMetrics(runsByExperiment?.[id])
  ]));
  const baselineAccuracy = finiteValue(metrics.E1?.[metric]);
  const strategies = Object.fromEntries(experimentIds
    .filter(id => id !== 'E1')
    .map(id => {
      const accuracy = finiteValue(metrics[id]?.[metric]);
      return [id, {
        accuracy,
        baselineAccuracy,
        difference: accuracy === null || baselineAccuracy === null
          ? null
          : accuracy - baselineAccuracy
      }];
    }));

  return { metric, baselineAccuracy, strategies };
}

export function calculateDatasetSummary(runsByExperiment, experimentIds) {
  const metrics = Object.fromEntries(experimentIds.map(id => [
    id,
    calculateExperimentMetrics(runsByExperiment?.[id])
  ]));
  const peakWinners = findWinners(metrics, 'bestValidationAccuracy', 'max');
  const efficiencyWinners = Object.fromEntries(
    Object.entries(EFFICIENCY_METRICS).map(([key, metric]) => [
      key,
      findWinners(metrics, metric, 'min')
    ])
  );
  const peakImprovements = calculateImprovement(
    runsByExperiment,
    experimentIds,
    'bestValidationAccuracy'
  );
  const singleAdaptationAccuracy = ['E2', 'E3']
    .map(id => finiteValue(metrics[id]?.bestValidationAccuracy))
    .filter(value => value !== null);
  const combinedAccuracy = finiteValue(metrics.E4?.bestValidationAccuracy);
  const e4VsBestSingle = combinedAccuracy === null || singleAdaptationAccuracy.length === 0
    ? null
    : combinedAccuracy - Math.max(...singleAdaptationAccuracy);

  return {
    metrics,
    peakWinners,
    efficiencyWinners,
    peakImprovements,
    e4VsBestSingle
  };
}

export function getEfficiencyPoints(rows, axisKey) {
  if (!Array.isArray(rows) || !['epoch', ...Object.keys(EFFICIENCY_METRICS)].includes(axisKey)) {
    return [];
  }

  return rows
    .map(row => ({
      x: finiteValue(row?.[axisKey]),
      y: finiteValue(row?.valAccuracy),
      epoch: finiteValue(row?.epoch)
    }))
    .filter(point => point.x !== null && point.y !== null && point.epoch !== null)
    .sort((left, right) => left.x - right.x || left.epoch - right.epoch);
}

export function cumulativeEndpoints(runsByExperiment, experimentIds) {
  const summaries = Object.fromEntries(experimentIds.map(id => [
    id,
    calculateExperimentMetrics(runsByExperiment?.[id])
  ]));

  return Object.fromEntries(Object.entries(EFFICIENCY_METRICS).map(([key, metric]) => [
    key,
    Object.fromEntries(experimentIds.map(id => [
      id,
      finiteValue(summaries[id]?.[metric])
    ]))
  ]));
}
