// Run: node test_gestures.mjs. No camera, microphone or API required.
import assert from 'node:assert/strict';
import {GestureGate, PinchGate, handIntent, poseCommand, readHands} from './assets/gesture_logic.mjs';

const close = (actual, expected, note) => assert.ok(Math.abs(actual - expected) < 1e-9, `${note}: ${actual} != ${expected}`);

// Wrist at origin, knuckles 0.2 below: palm width and index reach are measured from these.
const points = ({x = .5, y = .5, width = .1, gap = .05, curled = false} = {}) => {
  const p = Array.from({length: 21}, () => ({x: 0, y: 0}));
  p[0] = {x: 0, y: 0};
  p[5] = {x: 0, y: .2};
  p[17] = {x: width, y: .2};
  p[8] = curled ? {x: 0, y: .1} : {x, y};
  p[4] = {x: p[8].x + gap, y: p[8].y};
  return p;
};
const pose = (name = 'None', score = .9) => ({categoryName: name, score});
const intent = (opts = {}, confidence) => handIntent(points(opts), pose(opts.pose, opts.score), confidence);

// A hand must be present, complete and close enough to measure.
assert.equal(handIntent(undefined, pose()), null);
assert.equal(handIntent(points().slice(0, 20), pose()), null);
const broken = points(); broken[8] = {x: NaN, y: .5};
assert.equal(handIntent(broken, pose()), null);
assert.equal(intent({width: .02}), null); // Too far from the camera to trust.

// An open palm summons the controls; anything else keeps pointing.
assert.equal(intent({pose: 'Open_Palm'}).name, 'summon');
assert.equal(intent({pose: 'Open_Palm', score: .4}).name, 'pointer');
assert.equal(intent({pose: 'Open_Palm'}, .95).name, 'pointer');
assert.equal(intent().name, 'pointer');

// The camera is mirrored, so the cursor moves with the hand and stays on screen.
assert.equal(intent({x: .9, y: .9}).x, 0);
assert.equal(intent({x: .1, y: .1}).x, 1);
assert.equal(intent({x: .1, y: .1}).y, 0);
assert.equal(intent({x: .1, y: .9}).y, 1);
assert.equal(intent({x: -.4, y: 1.4}).x, 1);
assert.equal(intent({x: -.4, y: 1.4}).y, 1);

// Pinch distance is measured against palm width, so it survives changing camera distance.
close(intent({gap: .02, width: .1}).ratio, .2, 'near pinch');
close(intent({gap: .04, width: .2}).ratio, .2, 'same pinch, hand twice as close');
assert.equal(intent({curled: true}).ratio, 2); // A folded fist is not a pinch.

// Both hands stay live, each with its own side. The preview is mirrored.
const frame = (hands) => ({
  landmarks: hands.map(h => points(h)),
  gestures: hands.map(h => [pose(h.pose, h.score)]),
  handedness: hands.map(h => [{categoryName: h.model}]),
});
const both = readHands(frame([{model: 'Right', x: .8}, {model: 'Left', x: .2, pose: 'Open_Palm'}]));
assert.equal(both.length, 2);
assert.equal(both[0].side, 'left');
assert.equal(both[1].side, 'right');
assert.equal(both[1].name, 'summon');
assert.ok(both[0].x < both[1].x, 'the mirrored cursor of the left hand stays on the left');

// Two hands the model labels the same way must still get separate cursors.
const sameLabel = readHands(frame([{model: 'Right'}, {model: 'Right'}]));
assert.equal(sameLabel.length, 2);
assert.notEqual(sameLabel[0].side, sameLabel[1].side);

assert.equal(readHands({landmarks: []}).length, 0);
assert.equal(readHands({}).length, 0);
assert.equal(readHands(frame([{}, {}, {}])).length, 2); // Never track more than two.
// An unusable hand is dropped without stealing the other one's side.
const oneBad = readHands(frame([{width: .01}, {model: 'Left'}]));
assert.equal(oneBad.length, 1);
assert.equal(oneBad[0].side, 'right');

// Shortcut poses fire without pointing at anything, so the mouse stays optional.
const posed = (...names) => readHands(frame(names.map(pose => ({pose}))));
for (const [name, action] of Object.entries({Open_Palm:'summon', Closed_Fist:'finish', Thumb_Up:'accept', Thumb_Down:'cancel', ILoveYou:'plan'})) {
  assert.equal(poseCommand(posed(name)), action);
}
assert.equal(poseCommand(posed('Closed_Fist', 'Closed_Fist')), 'pause'); // Two fists still wins over the one-fist shortcut.
assert.equal(poseCommand(posed('Thumb_Up', 'Thumb_Up')), null); // Two hands only ever mean pause.
assert.equal(poseCommand(posed('Pointing_Up')), null); // Pointing drives the cursor instead.
assert.equal(poseCommand(posed('Victory')), null); // No longer mapped.
assert.equal(poseCommand([]), null);
assert.equal(poseCommand(readHands(frame([{pose: 'Thumb_Up', score: .4}]))), null); // Too unsure to act on.

// Each hand pinches independently: one selecting must not disturb the other.
const gates = {left: new PinchGate(), right: new PinchGate()};
for (const side of ['left', 'right']) { gates[side].update(1, null, 0); gates[side].update(1, null, 200); }
gates.left.update(.05, 0, 300);
assert.equal(gates.right.update(.05, 2, 400).target, 2);
assert.equal(gates.left.update(.05, 0, 1050).fire, true);
assert.equal(gates.right.update(.05, 2, 1050).fire, false); // Started later, still charging.
assert.equal(gates.right.update(.05, 2, 1150).fire, true);

const gate = new PinchGate();
// A pinch held from the first frame must not fire: fingers have to open first.
assert.equal(gate.update(.05, 1, 0).fire, false);
assert.equal(gate.update(.05, 1, 2000).fire, false);
gate.update(1, null, 2000);
assert.equal(gate.update(.05, 1, 2100).fire, false); // Opened for less than 150ms.
gate.update(1, null, 2100);
gate.update(1, null, 2300);
const started = gate.update(.05, 2, 2400);
assert.equal(started.fire, false);
assert.equal(started.target, 2);
close(gate.update(.05, 2, 2775).progress, .5, 'charge halfway');
assert.equal(gate.update(.05, 3, 3150).target, 2); // Target latches when the pinch starts.
assert.equal(gate.update(.05, 3, 3150).fire, false); // ...and fires only once.
gate.update(1, null, 3200); gate.update(1, null, 3400);
assert.equal(gate.update(.05, null, 3500).target, null);
assert.equal(gate.update(.05, null, 4300).fire, false); // Pinching off-target does nothing.

// Hysteresis: releasing past the threshold alone must not re-arm the next selection.
gate.reset();
gate.update(1, null, 0); gate.update(1, null, 200);
gate.update(.05, 0, 300);
gate.update(.3, 0, 400); // Between threshold and release: still the same pinch.
assert.equal(gate.update(.05, 1, 500).target, 0);
assert.equal(gate.update(null, null, 600).progress, 0); // Losing the hand cancels.

// Calibration: hold time and pinch sensitivity come from the sliders.
gate.reset();
gate.update(1, null, 0); gate.update(1, null, 200);
assert.equal(gate.update(.05, 3, 300, 1200).fire, false);
assert.equal(gate.update(.05, 3, 1400, 1200).fire, false);
assert.equal(gate.update(.05, 3, 1500, 1200).fire, true);
gate.reset();
gate.update(1, null, 0); gate.update(1, null, 200);
assert.equal(gate.update(.3, 4, 300, 750, .35).target, 4); // Looser pinch accepted at 35%.

// The deck is summoned by holding a palm for 500ms, the hold hands.js passes.
const summon = new GestureGate();
assert.equal(summon.update('summon', 0, 500).fire, false);
assert.equal(summon.update('summon', 499, 500).fire, false);
assert.equal(summon.update('summon', 500, 500).fire, true);
assert.equal(summon.update('summon', 3000, 500).fire, false); // A held palm never repeats.
summon.update(null, 3100, 500);
summon.update('summon', 3200, 500);
assert.equal(summon.update('summon', 4000, 500).fire, false); // Brief tracking loss does not repeat.
summon.update(null, 4100, 500); summon.update(null, 4500, 500);
summon.update('summon', 4600, 500);
assert.equal(summon.update('summon', 5100, 500).fire, true);

console.log('PASS: hand validation, summon vs pointer, mirrored cursor, scale-free pinch, two independent hands, pose shortcuts, arming, target latch, calibration');
