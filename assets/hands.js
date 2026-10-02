(async () => {
const root = element;
const ref = name => root.querySelector(`[data-ref="${name}"]`);
const button = name => root.querySelector(`[data-action="${name}"]`);
const base = props.assets_url;
const {GestureGate, PinchGate, readHands, poseCommand} = await import(`${base}/gesture_logic.mjs`);
const gate = new GestureGate();
const SIDES = ['left','right'];
// Each hand keeps its own cursor and pinch so the two never interfere.
const pinchGates = {left: new PinchGate(), right: new PinchGate()};
const smooth = {left:null, right:null}, trails = {left:[], right:[]};
let lastHandSeen = 0;
let recognizer, enginePromise, stream, recorder;
let frame = 0, lastVideoTime = -1, generation = 0, recordingEpoch = 0;
let running = false, starting = false, recording = false, paused = false;
let busy = false, stage = '', pendingRequest = '', suppressSpeech = false;
let deckOpen = true, planId = '', hasPlan = false, speaking = false;
let audioContext, analyser, audioSource, audioUrl, startTime = 0, timer;
const MAGNET = 26;
const links = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
const status = text => { ref('status').textContent = text; };
function trace(text) {
  const list=ref('trace');
  if(list.lastElementChild?.textContent===text)return;
  const item=document.createElement('li');item.textContent=text;list.append(item);
  while(list.children.length>5)list.firstElementChild.remove();
  list.hidden=false;
}
const visible = () => root.isConnected && root.getClientRects().length > 0;
const hasDraft = () => Boolean(ref('transcript').value.trim());

function render() {
  const phase = paused ? 'paused' : recording ? 'listening' : busy ? 'thinking' : speaking ? 'speaking' : hasPlan || hasDraft() ? 'review' : 'idle';
  ref('reactor').dataset.state=phase;
  ref('phase').textContent={paused:'손동작 멈춤',listening:'듣는 중',thinking:'처리 중',speaking:'말하는 중',review:'확인해 주세요',idle:'대기 중'}[phase];
  ref('mode').textContent = paused ? '멈춤' : recording ? '녹음 중' : busy ? '처리 중' : running ? '준비됨' : '꺼짐';
  button('start').hidden = running;
  button('start').disabled = starting;
  button('stop').hidden = !running && !starting;
  button('pause').hidden = !running;
  button('pause').textContent = paused ? '손동작 다시 시작' : '손동작 멈춤';
  button('voice').disabled = starting || running;
  button('listen').disabled = !running || recording || busy;
  button('finish').disabled = !recording;
  button('accept').disabled = (!hasDraft() && !hasPlan) || busy || recording;
  button('read').disabled = busy || recording;
  button('plan').disabled = busy || recording;
  button('confirm-plan').disabled = busy || !hasPlan;
  ref('deck').hidden=!deckOpen;
  ref('voice-label').textContent=recording?'녹음 종료':'말하기';
  root.querySelector('[data-command="0"]').disabled=busy||recording;
  root.querySelector('[data-command="1"]').disabled=busy||!running;
  root.querySelector('[data-command="2"]').disabled=(!hasDraft()&&!hasPlan)||busy||recording;
  ref('draft').hidden = !hasDraft();
}

function clearDraft() {
  ref('transcript').value = '';
  render();
}

function stopSpeech() {
  ref('audio').pause();
  ref('audio').currentTime = 0;
  suppressSpeech = true;
  speaking=false;
}

function finishRecording(discard = false) {
  if (!recording) return;
  recording = false;
  clearInterval(timer);
  if (recorder && recorder.state !== 'inactive') {
    recorder.cancelled = discard;
    recorder.stop();
    if (!discard) status('말씀을 문장으로 바꾸고 있어요.');
  }
  stream?.getAudioTracks().forEach(track => { track.enabled = false; });
  ref('volume').style.width = '0%';
  render();
}

function stopAll() {
  generation++;
  recordingEpoch++;
  finishRecording(true);
  running = starting = paused = false;
  cancelAnimationFrame(frame);
  stream?.getTracks().forEach(track => track.stop());
  stream = null;
  ref('camera').srcObject = null;
  audioSource?.disconnect();
  analyser = null;
  if (audioContext) { audioContext.close().catch(() => {}); audioContext = null; }
  stopSpeech();
  if (!['execute','confirm_plan'].includes(stage)) { busy = false; pendingRequest = ''; }
  clearDraft();
  gate.reset();
  for(const side of SIDES){pinchGates[side].reset();smooth[side]=null;trails[side]=[];ref(`cursor-${side}`).hidden=true;}
  root.querySelectorAll('[data-command]').forEach(b=>b.classList.remove('selected'));
  deckOpen=true;dismissPlan();
  ref('preview').hidden = true;
  ref('progress').style.width = '0%';
  if (audioUrl) { URL.revokeObjectURL(audioUrl); audioUrl = null; }
  ref('audio').removeAttribute('src');
  ref('audio').hidden = true;
  status(['execute','confirm_plan'].includes(stage) && busy ? '기기는 꺼졌어요. 이미 실행한 작업은 계속 처리되며 결과는 대화에 남습니다.' : '카메라·마이크·화면 연결을 모두 껐어요.');
  render();
}

async function prepareEngine() {
  if (recognizer) return recognizer;
  if (!enginePromise) enginePromise = (async () => {
    const {GestureRecognizer, FilesetResolver} = await import(`${base}/mediapipe/vision_bundle.mjs`);
    const files = await FilesetResolver.forVisionTasks(`${base}/mediapipe/wasm`);
    const options = {baseOptions: {modelAssetPath: `${base}/mediapipe/gesture_recognizer.task`, delegate:'GPU'}, runningMode:'VIDEO', numHands:2};
    // ponytail: falls back to CPU if this machine has no usable GPU delegate.
    try { recognizer = await GestureRecognizer.createFromOptions(files, options); return recognizer; }
    catch { options.baseOptions.delegate = 'CPU'; }
    recognizer = await GestureRecognizer.createFromOptions(files, options);
    return recognizer;
  })().catch(error => { enginePromise = null; throw error; });
  return enginePromise;
}

async function startDevices(camera = true) {
  if (starting || running || busy) return;
  stopAll();
  const run = ++generation;
  starting = true;
  status(camera ? '손 인식 모델을 준비하고 있어요. 처음에는 잠시 걸릴 수 있어요.' : '마이크 사용을 허용해 주세요.');
  render();
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('unsupported');
    audioContext = new AudioContext();
    await audioContext.resume();
    if (camera) await prepareEngine();
    if (run !== generation) return;
    status('브라우저의 카메라·마이크 사용 요청을 허용해 주세요.');
    const devices = await navigator.mediaDevices.getUserMedia({video: camera ? {width:{ideal:640},height:{ideal:480},facingMode:'user'} : false, audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true}});
    if (run !== generation) { devices.getTracks().forEach(t=>t.stop()); return; }
    stream = devices;
    stream.getTracks().forEach(t => t.addEventListener('ended', () => { if (running) stopAll(); }));
    stream.getAudioTracks().forEach(t => { t.enabled = false; });
    audioSource = audioContext.createMediaStreamSource(new MediaStream(stream.getAudioTracks()));
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;
    audioSource.connect(analyser);
    if (camera) {
      ref('camera').srcObject = new MediaStream(stream.getVideoTracks());
      await ref('camera').play();
      if (run !== generation) return;
      ref('preview').hidden = false;
    }
    running = true;
    starting = false;
    lastVideoTime = -1;
    status(camera ? '준비됐어요. 양손 모두 커서가 됩니다. 손끝으로 ‘말하기’를 가리킨 뒤 엄지와 검지를 집으세요.' : '마이크가 준비됐어요. ‘말하기’를 눌러 시작하세요.');
    render();
    frame = requestAnimationFrame(tick);
  } catch (error) {
    if (run !== generation) return;
    stopAll();
    const messages = {NotAllowedError:'카메라·마이크 권한이 필요해요. 브라우저 설정에서 허용 후 다시 켜 주세요.',NotFoundError:'카메라 또는 마이크를 찾지 못했어요. 연결을 확인하거나 마이크만 켜기를 사용하세요.',NotReadableError:'다른 앱이 카메라·마이크를 사용 중일 수 있어요. 사용 중인 앱을 확인해 주세요.'};
    status(messages[error.name] || '기기를 시작하지 못했어요. Chrome에서 열거나 기기 연결·손 인식 파일을 확인해 주세요.');
  }
}

function draw(result, hands) {
  const canvas = ref('canvas'), ctx = canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.strokeStyle = '#5ad8ee';
  ctx.fillStyle = '#b6f8ff'; ctx.lineWidth = 3;
  for (const points of result.landmarks || []) {
    for (const [a,b] of links) { ctx.beginPath(); ctx.moveTo((1-points[a].x)*640,points[a].y*480);ctx.lineTo((1-points[b].x)*640,points[b].y*480);ctx.stroke(); }
    for (const p of points) {ctx.beginPath();ctx.arc((1-p.x)*640,p.y*480,4,0,Math.PI*2);ctx.fill();}
  }
  for (const side of SIDES) {
    const hand = hands.find(h => h.side === side);
    if (!hand) { trails[side] = []; continue; }
    ctx.fillStyle = side === 'right' ? '#b8f7c9' : '#b6f8ff';
    trails[side] = [...trails[side], [hand.x*640, hand.y*480]].slice(-10);
    trails[side].forEach(([x,y],i)=>{ctx.globalAlpha=(i+1)/trails[side].length*.55;ctx.beginPath();ctx.arc(x,y,3+i*.8,0,Math.PI*2);ctx.fill();});
    ctx.globalAlpha=1;
  }
}

function tick(time) {
  if (!running) return;
  if (!visible() || document.hidden) {stopAll();return;}
  if (recording && analyser) {
    const values = new Float32Array(analyser.fftSize);
    analyser.getFloatTimeDomainData(values);
    const rms = Math.sqrt(values.reduce((n,v)=>n+v*v,0)/values.length);
    ref('volume').style.width = `${Math.min(100,rms*700)}%`;
  }
  const video = ref('camera');
  // ponytail: main-thread inference capped at 12fps; use a worker if low-end devices stutter.
  // Run on every fresh camera frame: the cursor lags by one frame at most.
  if (stream?.getVideoTracks().length && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    try {
      const result = recognizer.recognizeForVideo(video, time);
      const hands = readHands(result, Number(ref('confidence').value)/100);
      const command = poseCommand(hands);
      const shortcut = gate.update(command, time, command === 'pause' ? 1000 : command === 'summon' ? 500 : 800);
      if (paused) {
        if (shortcut.fire && command === 'pause') togglePause(false);
      } else {
        spatialInput(hands,time);
        draw(result,hands);
        if (hands.length) lastHandSeen = time;
        if (shortcut.fire) runPose(command);
        if (deckOpen && !hands.length && time-lastHandSeen > 1500) {deckOpen=false;render();}
      }

    } catch { stopAll(); status('손 인식이 중단됐어요. 기기를 다시 켜거나 동작 연습을 사용해 주세요.'); return; }
  }
  frame = requestAnimationFrame(tick);
}

async function record() {
  if (!running || busy || recording) return;
  if (ref('transcript').value.trim()) {status('확인 중인 문장을 먼저 실행하거나 취소해 주세요.');return;}
  stopSpeech();
  {
    const mime = ['audio/webm;codecs=opus','audio/mp4','audio/ogg;codecs=opus'].find(t=>window.MediaRecorder?.isTypeSupported(t));
    if (!mime) {status('이 브라우저에서는 녹음을 지원하지 않아요. Chrome에서 열어 주세요.');return;}
    const chunks = [], epoch = ++recordingEpoch;
    stream.getAudioTracks().forEach(t=>{t.enabled=true;});
    let rec;
    try {rec = new MediaRecorder(new MediaStream(stream.getAudioTracks()),{mimeType:mime});}
    catch {stream.getAudioTracks().forEach(t=>{t.enabled=false;});status('마이크 녹음을 시작하지 못했어요. 기기를 다시 연결해 주세요.');return;}
    recorder = rec;
    rec.ondataavailable = event => {if(event.data.size) chunks.push(event.data);};
    rec.onerror = () => {recordingEpoch++;finishRecording(true);status('녹음이 중단됐어요. 다시 시작해 주세요.');};
    rec.onstop = async () => {
      if (rec.cancelled || epoch !== recordingEpoch) return;
      const blob = new Blob(chunks,{type:mime});
      if (blob.size < 100 || blob.size > 6000000) {status('녹음이 너무 짧거나 길어요. 다시 말씀해 주세요.');return;}
      try {const data = await toBase64(blob);if (epoch === recordingEpoch) request('transcribe',{audio:data,mime});}
      catch {status('녹음 내용을 읽지 못했어요. 다시 말씀해 주세요.');}
    };
    try {rec.start(250);} catch {stream.getAudioTracks().forEach(t=>{t.enabled=false;});status('녹음을 시작하지 못했어요. 기기를 다시 연결해 주세요.');return;}
  }
  recording = true; startTime = Date.now();
  status('듣고 있어요. 말이 끝나면 ‘말하기’를 다시 집으세요. (최대 45초)');
  timer = setInterval(()=>{
    if (!recording) return;
    const seconds = Math.floor((Date.now()-startTime)/1000);
    status(`듣고 있어요 · ${seconds} / 45초. ‘말하기’로 종료 · ‘취소’로 중단`);
    if (seconds >= 45) finishRecording();
  },1000);
  render();
}

function toBase64(blob) {
  return new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(String(reader.result).split(',')[1]);reader.onerror=reject;reader.readAsDataURL(blob);});
}

function request(action, payload={}) {
  if (busy) return;
  busy = true; stage = action; suppressSpeech = false;
  ref('trace').replaceChildren();trace({plan:'기록을 참고해 계획 설계',confirm_plan:'검토한 계획 등록',transcribe:'음성을 문장으로 변환',execute:'요청 해석 및 도구 실행',speak_last:'음성 답변 생성'}[action]);
  pendingRequest = crypto.randomUUID();
  render();
  status(action === 'transcribe' ? '말씀을 문장으로 바꾸고 있어요…' : '자비스가 요청을 처리하고 있어요…');
  trigger('command',{request_id:pendingRequest,action,...payload});
}

function accept() {
  if (busy || recording) return;
  if(hasPlan){confirmPlan();return;}
  if (!hasDraft()) {status('먼저 말씀하거나 화면을 가져와 주세요.');return;}
  if(ref('plan-mode').checked){designPlan();return;}
  const text = ref('transcript').value.trim() || '첨부 화면의 핵심 내용을 한국어로 간결하게 요약해 줘.';
  request('execute',{text,speak:ref('speak').checked});
}

function cancel() {
  recordingEpoch++;
  finishRecording(true); stopSpeech();
  if (busy && ['execute','confirm_plan'].includes(stage)) {
    status('음성을 중단했어요. 이미 실행한 작업은 계속 처리됩니다. 대화와 처리 내역을 확인해 주세요.');
  } else {
    busy = false; pendingRequest = ''; stage = '';
    clearDraft();status('녹음과 대기 중인 요청을 취소했어요.');
  }
  deckOpen=true;dismissPlan();
  render();
}

function perform(action) {
  ({listen:record,finish:()=>finishRecording(),accept,cancel})[action]?.();
}

function firePulse(px,py) {
  const pulse=ref('pulse');pulse.hidden=false;pulse.style.left=`${px}px`;pulse.style.top=`${py}px`;
  pulse.getAnimations().forEach(a=>a.cancel());
  if(matchMedia('(prefers-reduced-motion: reduce)').matches){pulse.hidden=true;return;}
  pulse.animate([{transform:'translate(-50%,-50%) scale(.3)',opacity:1},{transform:'translate(-50%,-50%) scale(3)',opacity:0}],{duration:500}).finished.then(()=>{pulse.hidden=true;}).catch(()=>{});
}

function spatialInput(hands,time,mouse=false) {
  const candidates=Array.from(root.querySelectorAll('[data-command]'));
  const rect=ref('surface').getBoundingClientRect();
  const hovered=new Set();
  let progress=0;
  for(const side of SIDES){
    const hand=hands.find(h=>h.side===side), cursor=ref(`cursor-${side}`);
    if(!hand){pinchGates[side].update(null,null,time);smooth[side]=null;cursor.hidden=true;continue;}
    if(!smooth[side]||mouse)smooth[side]={x:hand.x,y:hand.y};
    else {smooth[side].x+=(hand.x-smooth[side].x)*.55;smooth[side].y+=(hand.y-smooth[side].y)*.55;}
    const px=smooth[side].x*rect.width,py=smooth[side].y*rect.height;
    const cx=px+rect.left, cy=py+rect.top;
    const target=candidates.find(b=>{const r=b.getBoundingClientRect();return !b.disabled&&cx>=r.left-MAGNET&&cx<=r.right+MAGNET&&cy>=r.top-MAGNET&&cy<=r.bottom+MAGNET;});
    const held=pinchGates[side].update(hand.ratio,target?Number(target.dataset.command):null,time,Number(ref('hold').value),Number(ref('pinch').value)/100);
    const focus=held.target??(target?Number(target.dataset.command):null);
    if(focus!==null)hovered.add(focus);
    progress=Math.max(progress,held.progress);
    cursor.hidden=false;
    cursor.style.transform=`translate(${px}px,${py}px)`;
    cursor.dataset.locked=held.progress>0?'true':'false';
    cursor.querySelector('.cursor-charge').style.strokeDashoffset=String(126*(1-held.progress));
    if(held.fire){
      const chosen=root.querySelector(`[data-command="${held.target}"]`);
      if(chosen&&!chosen.disabled){firePulse(px,py);chooseCommand(held.target);}
    }
  }
  candidates.forEach(b=>b.classList.toggle('selected',hovered.has(Number(b.dataset.command))));
  ref('tracking').textContent=!hands.length?'손을 보여 주세요':!hovered.size?'메뉴를 가리키세요':progress>0?'선택하는 중':'집으면 선택돼요';
  ref('progress').style.width=`${progress*100}%`;
  root.querySelector('[role="progressbar"]').setAttribute('aria-valuenow',String(Math.round(progress*100)));
  return {progress};
}

watch('value', () => {
  let data;
  try {data=JSON.parse(props.value);} catch {return;}
  if (!data.request_id || data.request_id !== pendingRequest) return;
  if (data.event === 'working') {trace(data.text);if(!suppressSpeech)status(data.text);return;}
  const completedStage=stage;
  busy=false;stage='';pendingRequest='';
  if(data.event==='plan') {
    showPlan(data.plan,data.plan_id);trace('계획 검증 완료 · 사용자 검토 대기');
  } else if (data.event === 'transcript') {
    ref('transcript').value=data.text;
    status('이렇게 들었어요. 문장을 확인한 뒤 ‘확인 후 실행’을 집으면 실행합니다. 수정하거나 다시 녹음해도 좋아요.');
  } else if (data.event === 'done') {
    if(completedStage!=='speak_last'){dismissPlan();clearDraft();}
    trace('처리 완료');status(data.warning || '요청을 처리했어요. 대화와 기록에서 확인할 수 있어요.');
    if(data.audio && !suppressSpeech && ref('speak').checked) {
      if(audioUrl)URL.revokeObjectURL(audioUrl);
      const bytes=Uint8Array.from(atob(data.audio),c=>c.charCodeAt(0));
      audioUrl=URL.createObjectURL(new Blob([bytes],{type:'audio/mpeg'}));
      ref('audio').src=audioUrl;ref('audio').hidden=false;
      ref('audio').play().catch(()=>status('음성 답변이 준비됐어요. 아래 재생 버튼을 눌러 주세요.'));
    }
  } else if(data.event === 'error') {status(data.text);trace(data.text);}
  render();
});

root.addEventListener('click', event => {
  const target=event.target.closest('button');
  if (!target || !root.contains(target)) return;
  if(target.dataset.command!==undefined){chooseCommand(Number(target.dataset.command));return;}
  const extra={plan:designPlan,'confirm-plan':confirmPlan,'dismiss-plan':dismissPlan};
  if(extra[target.dataset.action]){extra[target.dataset.action]();return;}
  const actions={start:()=>startDevices(),stop:stopAll,voice:()=>startDevices(false),pause:()=>togglePause(),read:()=>{if(!busy)request('speak_last');},listen:()=>perform('listen'),finish:()=>perform('finish'),accept:()=>perform('accept'),cancel};
  actions[target.dataset.action]?.();
});
function togglePause(force) {
  if (!running) return;
  paused = force ?? !paused;
  if (paused) {
    spatialInput([], performance.now());
    ref('canvas').getContext('2d').clearRect(0,0,640,480);
    ref('tracking').textContent = '손동작을 멈췄어요';
  }
  status(paused ? '손동작을 멈췄어요. 버튼과 키보드는 그대로 쓸 수 있어요. 양 주먹을 쥐면 다시 시작합니다.'
                : '손동작을 다시 시작했어요.');
  render();
}

function runPose(name) {
  if (name === 'pause') {togglePause();return;}
  if (name === 'summon') {
    if (deckOpen) return;
    deckOpen=true;render();status('컨트롤을 불러왔어요. 손끝으로 가리킨 뒤 집으세요.');
    return;
  }
  if (name === 'plan') {
    if (busy||recording) return;
    ref('plan-mode').checked=true;ref('mission-editor').open=true;
    status('목표 설계를 켰어요. ‘말하기’를 집어서 목표를 말씀하세요.');
    return;
  }
  if (name === 'finish') {
    if (recording) finishRecording();  // A one-hand fist mid-recording ends it; otherwise it's a no-op.
    return;
  }
  ({accept, cancel})[name]?.();
}

function chooseCommand(index) {
  deckOpen=true;render();
  if(index===0){request('execute',{text:'오늘 일정과 남은 할 일을 조회해서 먼저 할 일 3가지를 추천해 줘.',speak:ref('speak').checked});}
  if(index===1){if(recording)finishRecording();else record();}
  if(index===2)accept();
  if(index===3)cancel();
}
function dismissPlan(){planId='';hasPlan=false;ref('mission').hidden=true;render();}
function showPlan(plan,id) {
  clearDraft();planId=id;hasPlan=plan.items.length>0;
  ref('plan-summary').textContent=plan.summary;ref('plan-caution').textContent=plan.caution||'';
  ref('plan-items').replaceChildren();
  for(const item of plan.items){const li=document.createElement('li');const title=document.createElement('strong');title.textContent=item.title;const detail=document.createElement('p');detail.textContent=`${{task:'할 일',event:'일정',note:'메모'}[item.kind]} · ${item.due||'날짜 미정'}${item.details?' · '+item.details:''}`;li.append(title,detail);ref('plan-items').append(li);}
  ref('mission').hidden=false;
  status(hasPlan?'계획을 확인하세요. 검토 후 ‘확인 후 실행’을 가리키고 집으면 표시된 항목 전체를 등록해요.':'계획을 만들기 위해 추가 정보가 필요해요. 질문을 확인하고 목표를 보완해 주세요.');render();
}
function designPlan() {
  if(busy||recording)return;
  const text=ref('transcript').value.trim()||ref('goal').value.trim();
  if(!text){status('설계할 목표를 입력하거나 말씀해 주세요.');ref('goal').focus();return;}
  dismissPlan();request('plan',{text});
}
function confirmPlan() {
  if(busy||!hasPlan)return;
  request('confirm_plan',{plan_id:planId});
}
ref('audio').addEventListener('play',()=>{speaking=true;render();});
ref('audio').addEventListener('pause',()=>{speaking=false;render();});
ref('audio').addEventListener('ended',()=>{speaking=false;render();});
root.addEventListener('keydown',event=>{
  if(event.key==='Escape')cancel();
});
ref('transcript').addEventListener('input',render);
ref('hold').addEventListener('input',()=>{ref('hold-label').textContent=`${(Number(ref('hold').value)/1000).toFixed(2)}초`;gate.reset();});
ref('pinch').addEventListener('input',()=>{ref('pinch-label').textContent=`${ref('pinch').value}%`;for(const side of SIDES)pinchGates[side].reset();});
ref('confidence').addEventListener('input',()=>{ref('confidence-label').textContent=`${ref('confidence').value}%`;gate.reset();});
ref('speak').addEventListener('change',()=>{if(!ref('speak').checked)stopSpeech();});
const visibilityHandler=()=>{if(document.hidden)stopAll();};
window.addEventListener('pagehide',stopAll,{once:true});
document.addEventListener('visibilitychange',visibilityHandler);
const observer=new MutationObserver(()=>{
  if(!root.isConnected){stopAll();recognizer?.close();observer.disconnect();document.removeEventListener('visibilitychange',visibilityHandler);}
  else if(running&&!visible())stopAll();
});
observer.observe(document.body,{subtree:true,childList:true,attributes:true,attributeFilter:['style','hidden']});
render();
})().catch(() => {
  element.querySelector('[data-ref="status"]').textContent = '손동작 화면을 준비하지 못했어요. 새로고침 후 다시 시도해 주세요.';
});
