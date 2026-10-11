import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import {
  calculateDatasetSummary,
  calculateImprovement,
  cumulativeEndpoints,
  getEfficiencyPoints
} from '../src/utils/crossDatasetAnalysis.js';

const experimentIds = ['E1', 'E2', 'E3', 'E4'];
const dataRoot = fileURLToPath(new URL('../public/data/', import.meta.url));

function parseRun(csv) {
  const [header, ...lines] = csv.trim().split(/\r?\n/);
  const headers = header.split(',');

  return lines.map(line => {
    const values = line.split(',');
    const value = key => Number(values[headers.indexOf(key)]);
    return {
      epoch: value('epoch'),
      valAccuracy: value('val_accuracy'),
      trainAccuracy: value('train_accuracy'),
      valLoss: value('val_loss'),
      optimizerUpdates: value('optimizer_updates'),
      elapsedSeconds: value('elapsed_seconds')
    };
  });
}

async function loadRuns(dataset, epochs) {
  return Object.fromEntries(await Promise.all(experimentIds.map(async id => {
    const file = path.join(dataRoot, dataset, `${epochs}epochs`, `${id.toLowerCase()}.csv`);
    const rows = parseRun(await readFile(file, 'utf8'));
    return [id, rows];
  })));
}

test('improvements support best and final accuracy, negative gains, and ties', () => {
  const runs = {
    E1: [{ epoch: 1, valAccuracy: 80 }, { epoch: 2, valAccuracy: 82 }],
    E2: [{ epoch: 1, valAccuracy: 81 }, { epoch: 2, valAccuracy: 80 }],
    E3: [{ epoch: 1, valAccuracy: 82 }, { epoch: 2, valAccuracy: 82 }],
    E4: [{ epoch: 1, valAccuracy: 79 }, { epoch: 2, valAccuracy: 80 }]
  };

  const best = calculateImprovement(runs, experimentIds, 'bestValidationAccuracy');
  const final = calculateImprovement(runs, experimentIds, 'finalValidationAccuracy');

  assert.equal(best.strategies.E2.difference, -1);
  assert.equal(best.strategies.E3.difference, 0);
  assert.equal(best.strategies.E4.difference, -2);
  assert.equal(final.strategies.E2.difference, -2);
  assert.equal(final.strategies.E3.difference, 0);
});

test('summary retains ties and reports invalid cumulative series as unavailable', () => {
  const tiedRuns = Object.fromEntries(experimentIds.map(id => [id, [
    { epoch: 1, valAccuracy: 80, optimizerUpdates: 5, elapsedSeconds: 2 }
  ]]));
  const tied = calculateDatasetSummary(tiedRuns, experimentIds);

  assert.deepEqual(tied.peakWinners, experimentIds);
  assert.deepEqual(tied.efficiencyWinners.optimizerUpdates, experimentIds);
  assert.deepEqual(tied.efficiencyWinners.elapsedSeconds, experimentIds);

  tiedRuns.E2.push({ epoch: 2, valAccuracy: 81, optimizerUpdates: 4, elapsedSeconds: 3 });
  const endpoints = cumulativeEndpoints(tiedRuns, experimentIds);
  assert.equal(endpoints.optimizerUpdates.E2, null);
  assert.equal(endpoints.elapsedSeconds.E2, 3);
});

test('efficiency points preserve recorded coordinates, omit invalid observations, and sort numerically', () => {
  const points = getEfficiencyPoints([
    { epoch: 3, valAccuracy: 83, optimizerUpdates: 90, elapsedSeconds: 9 },
    { epoch: 1, valAccuracy: 80, optimizerUpdates: 10, elapsedSeconds: 1 },
    { epoch: 2, valAccuracy: 82, optimizerUpdates: 50, elapsedSeconds: Number.NaN }
  ], 'optimizerUpdates');

  assert.deepEqual(points, [
    { x: 10, y: 80, epoch: 1 },
    { x: 50, y: 82, epoch: 2 },
    { x: 90, y: 83, epoch: 3 }
  ]);
  assert.deepEqual(getEfficiencyPoints([], 'elapsedSeconds'), []);
});

test('all recorded configurations match direct CSV-derived metrics', async () => {
  for (const [dataset, epochs] of [['cifar10', 50], ['cifar10', 80], ['cifar100', 120]]) {
    const runs = await loadRuns(dataset, epochs);
    const summary = calculateDatasetSummary(runs, experimentIds);
    const endpoints = cumulativeEndpoints(runs, experimentIds);

    for (const id of experimentIds) {
      const rows = runs[id];
      const best = Math.max(...rows.map(row => row.valAccuracy));
      const last = rows.at(-1);
      const metric = summary.metrics[id];

      assert.equal(rows.length, epochs);
      assert.deepEqual(metric.bestValidationEpochs, rows
        .filter(row => row.valAccuracy === best)
        .map(row => row.epoch));
      assert.equal(metric.bestValidationAccuracy, best);
      assert.equal(metric.finalValidationAccuracy, last.valAccuracy);
      assert.equal(endpoints.optimizerUpdates[id], last.optimizerUpdates);
      assert.equal(endpoints.elapsedSeconds[id], last.elapsedSeconds);
      assert.ok(rows.every((row, index) => index === 0 || (
        row.optimizerUpdates >= rows[index - 1].optimizerUpdates &&
        row.elapsedSeconds >= rows[index - 1].elapsedSeconds
      )));
    }

    const improvements = calculateImprovement(runs, experimentIds, 'bestValidationAccuracy');
    for (const id of ['E2', 'E3', 'E4']) {
      assert.equal(
        improvements.strategies[id].difference,
        summary.metrics[id].bestValidationAccuracy - summary.metrics.E1.bestValidationAccuracy
      );
    }
  }
});
