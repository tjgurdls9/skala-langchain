// Spatial input: each hand drives its own pointer; a palm reveals the controls.
export function handIntent(points, gesture, confidence = .65) {
  if (points?.length !== 21 || points.some(v => !Number.isFinite(v.x) || !Number.isFinite(v.y))) return null;
  const distance = (a,b) => Math.hypot(points[a].x-points[b].x,points[a].y-points[b].y);
  const width = distance(5,17);
  if (width < .025) return null;
  const pose = gesture?.score >= confidence ? gesture.categoryName : null;
  // A folded fist is not an intentional pinch. Palm width normalizes camera distance.
  const reaches = distance(0,8) > distance(0,5) * .85;
  const ratio = reaches ? distance(4,8)/width : 2;
  return {name:pose==='Open_Palm'?'summon':'pointer', pose,
    x:Math.max(0,Math.min(1,(1-points[8].x-.1)/.8)),
    y:Math.max(0,Math.min(1,(points[8].y-.1)/.8)), ratio};
}

// Both hands stay live. Each keeps a stable side so its cursor and pinch never swap.
export function readHands(result, confidence = .65) {
  const hands = [], taken = new Set();
  const frames = (result.landmarks || []).slice(0,2);
  for (let i = 0; i < frames.length; i++) {
    const intent = handIntent(frames[i], result.gestures?.[i]?.[0], confidence);
    if (!intent) continue;
    // The preview is mirrored, so the model's "Left" is the hand shown on the right.
    let side = result.handedness?.[i]?.[0]?.categoryName === 'Left' ? 'right' : 'left';
    if (taken.has(side)) side = side === 'left' ? 'right' : 'left';
    if (taken.has(side)) continue;
    taken.add(side);
    hands.push({...intent, side});
  }
  return hands;
}

// Shortcuts: a pose fires its command outright, so nothing needs pointing at.
const POSES = {Open_Palm:'summon', Closed_Fist:'finish', Thumb_Up:'accept', Thumb_Down:'cancel', ILoveYou:'plan'};
export function poseCommand(hands) {
  if (hands.length === 2) return hands.every(h => h.pose === 'Closed_Fist') ? 'pause' : null;
  if (hands.length !== 1) return null;
  return POSES[hands[0].pose] ?? null;
}

export class PinchGate {
  constructor(){this.reset();}
  reset(){this.ready=false;this.openSince=null;this.start=null;this.target=null;this.fired=false;}
  update(ratio,target,time,hold=750,threshold=.25){
    if(ratio===null){this.reset();return {fire:false,progress:0,target:null};}
    if(ratio>threshold+.12){
      this.start=null;this.target=null;this.fired=false;
      this.openSince??=time;
      if(time-this.openSince>=150)this.ready=true;
      return {fire:false,progress:0,target:null};
    }
    this.openSince=null;
    if(ratio<threshold && this.ready && this.start===null){
      this.ready=false;this.start=time;this.target=target;this.fired=false;
    }
    const progress=this.start===null?0:Math.min(1,(time-this.start)/hold);
    const fire=this.target!==null&&progress===1&&!this.fired;
    if(fire)this.fired=true;
    return {fire,progress,target:this.target};
  }
}

export class GestureGate {
  constructor(){this.reset();}
  reset(){this.candidate='';this.since=0;this.latched='';this.absentSince=null;}
  update(name,time,hold=750){
    if(!name){this.candidate='';this.absentSince??=time;if(time-this.absentSince>=350)this.latched='';return {fire:false,progress:0};}
    this.absentSince=null;
    if(this.candidate!==name){this.candidate=name;this.since=time;}
    const progress=Math.min(1,(time-this.since)/hold);
    if(this.latched===name||progress<1)return {fire:false,progress};
    this.latched=name;return {fire:true,progress:1};
  }
}
