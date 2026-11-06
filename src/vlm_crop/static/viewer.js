"use strict";
const data = window.EXPERIMENT;
const $ = id => document.getElementById(id);
const names = {single:"One look",repeat:"Look again",crop:"Choose a crop"};
const accents = {single:"#8a93ad",repeat:"#528cff",crop:"#12a68c"};
const conditions = ["single","repeat","crop"];
let timer = null;
function el(tag, text, cls) { const node=document.createElement(tag); if(text!==undefined) node.textContent=text; if(cls) node.className=cls; return node; }
function options(node, values) { node.replaceChildren(...values.map(([value,label]) => {const opt=el("option",label);opt.value=value;return opt;})); }
function stop(){clearInterval(timer);timer=null;$("play").textContent="Play replay";}
function modelEpisodes(){return data.episodes.filter(e=>e.model===$("model").value);}
function group(){return modelEpisodes().filter(e=>e.sample_id===$("example").value);}
function fillExamples(){
  stop();
  const episodes=modelEpisodes().filter(e=>e.task===$("task").value);
  let ids=[...new Set(episodes.map(e=>e.sample_id))];
  ids=ids.filter(id=>{const g=episodes.filter(e=>e.sample_id===id), c=g.find(e=>e.condition==="crop"), r=g.find(e=>e.condition==="repeat");
    if($("outcome").value==="all")return true;
    if($("outcome").value==="failure")return c&&!c.correct;
    if(!c||!r||c.status!=="completed"||r.status!=="completed")return false;
    return $("outcome").value==="improvement"?c.correct&&!r.correct:!c.correct&&r.correct;
  });
  options($("example"),ids.map((id,i)=>[id,`${i+1} · ${id.slice(0,8)}`]));
  $("empty").hidden=ids.length>0;$("replay").hidden=ids.length===0;
  if(ids.length) selectExample();
}
function selectModel(){
  options($("task"),[...new Set(modelEpisodes().map(e=>e.task))].map(t=>[t,t]));
  const summaries=data.stats.groups.filter(g=>g.model===$("model").value&&g.task==="All tasks");
  $("stats").replaceChildren(...conditions.map(c=>{const row=summaries.find(g=>g.condition===c), box=el("div",undefined,"stat");box.append(el("span",names[c]),el("strong",row?`${Math.round(row.accuracy*100)}%`:"—"),el("small",row?`${row.correct}/${row.n} correct · ${row.completed}/${row.n} completed`:"No episodes"));return box;}));
  fillExamples();
}
function selectExample(){
  stop();const g=group();$("round").max=Math.max(1,...g.map(e=>(e.views||[]).length));$("round").value=1;
  $("play").disabled=Number($("round").max)===1;
  $("question").textContent=g[0].question;$("truth").textContent=`Ground truth (evaluator only): ${g[0].groundtruth??"unavailable"}`;
  const exp=data.exports.find(e=>e.model===$("model").value&&e.sample_id===$("example").value);
  for(const type of ["png","gif"]){$(type).href=exp[type];$(type).setAttribute("download","");}
  render();
}
function render(){
  const g=group(), round=Number($("round").value);$("round-label").textContent=`${round} / ${$("round").max}`;
  $("panels").replaceChildren(...conditions.map(condition=>{
    const panel=el("article",undefined,"panel");panel.style.setProperty("--accent",accents[condition]);
    const head=el("header");head.append(el("h3",names[condition]));panel.append(head);
    const episode=g.find(e=>e.condition===condition);
    if(!episode?.views?.length){panel.append(el("p","No recorded view","answer"));return panel;}
    const index=Math.min(round,episode.views.length)-1,view=episode.views[index],done=round>=episode.views.length;
    head.append(el("p",`View ${index+1} of ${episode.views.length} recorded${done?" · finished":""}`,"subtitle"));
    const img=el("img",undefined,"view-image");img.src=view.image;img.alt=`Actual ${condition} input at view ${index+1}`;panel.append(img);
    const context=el("div",undefined,"context"),thumb=el("div",undefined,"thumbnail"),original=el("img");original.src=episode.views[0].image;original.alt="Original image with selected region";thumb.append(original);
    if(view.box){const [l,t,r,b]=view.box,box=el("div",undefined,"crop-box");Object.assign(box.style,{left:`${100*l/episode.width}%`,top:`${100*t/episode.height}%`,width:`${100*(r-l)/episode.width}%`,height:`${100*(b-t)/episode.height}%`});thumb.append(box);}
    context.append(thumb,el("p",view.box?`Model-selected region\n[${view.box.join(", ")}]`:(index===0?"Initial full image":"Full image requested again")));panel.append(context);
    const answer=el("div",undefined,"answer");answer.append(el("strong",done?`Answer: ${episode.answer??"—"}`:"Inspecting…"));
    if(done){const success=episode.status==="completed";answer.append(el("span",success?(episode.correct?"Correct":"Incorrect"):episode.status,`badge ${success?(episode.correct?"good":"bad"):""}`));answer.append(el("div",`${episode.input_tokens||0} input + ${episode.output_tokens||0} output tokens · ${(episode.latency_seconds||0).toFixed(1)}s · $${(episode.nominal_cost||0).toFixed(4)} nominal`,"usage"));}
    panel.append(answer);
    const details=el("details",undefined,"trace");details.append(el("summary","Recorded messages and actions"));
    details.append(el("pre",JSON.stringify((episode.turns||[]).filter(t=>t.view_round<=round),null,2)));panel.append(details);return panel;
  }));
}
options($("model"),[...new Set(data.episodes.map(e=>e.model))].map(m=>[m,m]));
$("scope").textContent=`${data.split==="pilot"?"Pilot smoke test — not benchmark evidence":"Exploratory evaluation"} · ${new Set(data.episodes.map(e=>e.sample_id)).size} unique image(s)`;
const mandatoryCrop = (data.config.protocol_version || 1) >= 2;
$("scope").textContent += mandatoryCrop ? " · mandatory crop" : " · optional crop (v1)";
if (mandatoryCrop) $("condition-description").textContent = "One look answers from one full image. Look again can request more full-image views. Choose a crop MUST inspect at least one model-selected region smaller than the full image before answering. Its first viewing action is forced to crop; the model chooses the coordinates. Both multi-view agents have up to five views including the initial image.";
$("method-note").textContent=data.stats.note;
$("model").addEventListener("change",selectModel);$("task").addEventListener("change",fillExamples);$("outcome").addEventListener("change",fillExamples);$("example").addEventListener("change",selectExample);$("round").addEventListener("input",()=>{stop();render();});
$("play").addEventListener("click",()=>{if(timer){stop();return;}if(Number($("round").value)>=Number($("round").max))$("round").value=1;render();$("play").textContent="Pause";timer=setInterval(()=>{$("round").value=Number($("round").value)+1;render();if(Number($("round").value)>=Number($("round").max))stop();},1400);});
selectModel();
