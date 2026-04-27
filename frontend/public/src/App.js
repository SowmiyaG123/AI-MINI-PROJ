import { useState, useEffect, useRef, useCallback } from "react";

const API = "http://localhost:8000";
const getSid = () => { let id = localStorage.getItem("rsid_v5"); if (!id) { id = crypto.randomUUID(); localStorage.setItem("rsid_v5", id); } return id; };
const C = { bg:"#0c0e0e",surf:"#141717",alt:"#1a1e1e",border:"#252929",gold:"#e8a027",goldD:"#b87c18",goldG:"rgba(232,160,39,0.14)",green:"#3ecf8e",red:"#f06565",blue:"#5b9cf6",purple:"#a78bfa",t1:"#f0ede8",t2:"#9a9590",t3:"#55514d",r:"13px" };
const ts = () => new Date().toLocaleTimeString([], { hour:"2-digit", minute:"2-digit" });
const md = t => <span dangerouslySetInnerHTML={{__html: t.replace(/\*\*(.*?)\*\*/g,`<strong style="color:${C.goldD}">$1</strong>`).replace(/\*(.*?)\*/g,`<em style="color:${C.t2}">$1</em>`).replace(/\n/g,"<br/>")}} />;

const Chip = ({ label, color="gold" }) => {
  const bg = color==="green"?"#173326":color==="blue"?"#172140":color==="red"?"#2a1515":color==="purple"?"#1e1535":"#2a1d08";
  const tc = color==="green"?C.green:color==="blue"?C.blue:color==="red"?C.red:color==="purple"?C.purple:C.gold;
  return <span style={{background:bg,color:tc,fontSize:10,fontWeight:700,padding:"2px 8px",borderRadius:20,textTransform:"uppercase",letterSpacing:"0.04em",whiteSpace:"nowrap"}}>{label}</span>;
};

function RecipeCard({ r, first }) {
  const [open, setOpen] = useState(first);
  const [tab, setTab] = useState("ingredients");
  const isAI = r._is_llm_generated || r.tags?.includes("ai-generated");
  const pc = r.match_pct >= 80 ? C.green : r.match_pct >= 50 ? C.gold : C.red;
  const matchedSet = new Set((r.matched_ingredients || []).map(x => x.toLowerCase()));
  return (
    <div style={{background:C.alt,border:`1px solid ${open?(isAI?C.purple:C.goldD):C.border}`,borderRadius:C.r,overflow:"hidden",marginBottom:8,boxShadow:open?`0 0 18px ${isAI?"rgba(167,139,250,0.15)":C.goldG}`:"none"}}>
      <button onClick={() => setOpen(v => !v)} style={{width:"100%",display:"flex",alignItems:"center",gap:10,padding:"12px 14px",background:"transparent",border:"none",cursor:"pointer",textAlign:"left"}}>
        <div style={{flex:1}}>
          <div style={{display:"flex",gap:6,alignItems:"center",flexWrap:"wrap",marginBottom:4}}>
            <span style={{fontSize:14,fontWeight:700,color:C.t1}}>{r.name}</span>
            {first && !isAI && <Chip label="Best Match" color="green"/>}
            {isAI && <Chip label="✨ AI Generated" color="purple"/>}
          </div>
          <div style={{display:"flex",gap:5,flexWrap:"wrap",alignItems:"center"}}>
            <Chip label={r.cuisine} color="blue"/>
            {r.diet?.map(d => <Chip key={d} label={d}/>)}
            <span style={{color:C.t3,fontSize:11}}>⏱ {r.time} · 👤 {r.servings} servings</span>
          </div>
        </div>
        {r.match_pct != null && <div style={{textAlign:"center",minWidth:46}}>
          <div style={{fontSize:18,fontWeight:800,color:pc}}>{r.match_pct}%</div>
          <div style={{fontSize:9,color:C.t3}}>match</div>
        </div>}
        <span style={{color:C.t3,fontSize:12}}>{open?"▲":"▼"}</span>
      </button>
      {r.match_pct != null && <div style={{padding:"0 14px 8px"}}>
        <div style={{background:C.border,borderRadius:4,height:4}}><div style={{height:"100%",width:`${r.match_pct}%`,background:pc,borderRadius:4,transition:"width .6s"}}/></div>
      </div>}
      {open && <>
        <div style={{display:"flex",borderTop:`1px solid ${C.border}`,borderBottom:`1px solid ${C.border}`}}>
          {["ingredients","steps","missing"].map(t2 => (
            <button key={t2} onClick={() => setTab(t2)} style={{flex:1,padding:"9px 0",fontSize:11,fontWeight:700,letterSpacing:"0.05em",textTransform:"uppercase",background:tab===t2?C.goldG:"transparent",color:tab===t2?C.gold:C.t3,border:"none",borderBottom:tab===t2?`2px solid ${C.gold}`:"2px solid transparent",cursor:"pointer"}}>{t2}</button>
          ))}
        </div>
        <div style={{padding:"12px 14px"}}>
          {tab==="ingredients" && <div style={{display:"flex",flexWrap:"wrap",gap:6}}>
            {Object.entries(r.ingredients).map(([k,v]) => {
              const isMatched = matchedSet.has(k.toLowerCase()) || [...matchedSet].some(m => k.toLowerCase().includes(m)||m.includes(k.toLowerCase()));
              return (
                <span key={k} style={{background:isMatched?"#173326":C.surf,border:`1px solid ${isMatched?C.green+"55":C.border}`,borderRadius:20,padding:"3px 10px",fontSize:12,color:C.t1}}>
                  <span style={{color:isMatched?C.green:C.gold,fontWeight:600}}>{k}</span>
                  <span style={{color:C.t3}}> · {v}</span>
                  {isMatched && <span style={{color:C.green,marginLeft:4,fontSize:10}}>✓</span>}
                </span>
              );
            })}
            {matchedSet.size > 0 && <div style={{width:"100%",marginTop:6,fontSize:11,color:C.green}}>✅ You have {matchedSet.size} of {Object.keys(r.ingredients).length} ingredients</div>}
          </div>}
          {tab==="steps" && <ol style={{listStyle:"none",padding:0,margin:0,display:"flex",flexDirection:"column",gap:10}}>
            {r.steps.map((s,i) => (
              <li key={i} style={{display:"flex",gap:10,alignItems:"flex-start"}}>
                <span style={{minWidth:24,height:24,background:C.gold,color:"#000",borderRadius:"50%",display:"flex",alignItems:"center",justifyContent:"center",fontSize:10,fontWeight:800,flexShrink:0}}>{i+1}</span>
                <span style={{fontSize:13,color:C.t2,lineHeight:1.7}}>{s}</span>
              </li>
            ))}
          </ol>}
          {tab==="missing" && (r.missing?.length > 0
            ? <><p style={{color:C.t3,fontSize:12,marginBottom:8}}>Missing these — ask me for substitutes!</p>
                <div style={{display:"flex",flexWrap:"wrap",gap:6}}>{r.missing.map(m=><span key={m} style={{background:"#2a1515",border:`1px solid ${C.red}44`,color:C.red,borderRadius:20,padding:"3px 10px",fontSize:12}}>⚠ {m}</span>)}</div></>
            : <p style={{color:C.green,fontSize:13}}>✅ You have all the ingredients!</p>)}
        </div>
      </>}
    </div>
  );
}

function SubCard({ sub }) {
  if (!sub) return null;
  return <div style={{background:"#141f17",border:`1px solid ${C.green}44`,borderRadius:C.r,padding:14,marginTop:8}}>
    <div style={{fontSize:12,fontWeight:700,color:C.green,marginBottom:10}}>🔄 Substitutes for "{sub.ingredient}"</div>
    {sub.options.map((o,i) => <div key={i} style={{background:C.alt,borderRadius:8,padding:"8px 12px",marginBottom:6,fontSize:13,color:C.t2}}><span style={{color:C.t1,fontWeight:600}}>#{i+1} </span>{o}</div>)}
  </div>;
}

const QUICK = ["I have chicken, rice, tomato","Recipe using fish","Strawberry dessert","Carrot cake","Substitute for paneer","No beef, pasta please","Prawn curry","Banana pancakes","Chocolate cake","Cauliflower soup","Random recipe","Vegetarian Indian curry"];

export default function App() {
  const [sid] = useState(getSid);
  const [msgs, setMsgs] = useState([{id:0,role:"bot",text:"👋 Hi! Tell me your ingredients and I'll find recipes.\n\n• **'no beef'** — exclude ingredients\n• **'include chicken'** — undo exclusions\n• **'reset preferences'** — clear all filters\n• **'random recipe'** — surprise me!",recipes:[],sub:null,time:ts()}]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [prefs, setPrefs] = useState({ exclude:[], diet:[] });
  const [status, setStatus] = useState("checking");
  const endRef = useRef(null);
  const fileRef = useRef(null);
  const taRef = useRef(null);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior:"smooth" }); }, [msgs, loading]);
  useEffect(() => { fetch(`${API}/health`).then(r=>r.ok?setStatus("online"):setStatus("error")).catch(()=>setStatus("offline")); }, []);

  const addMsg = (role, text, recipes=[], sub=null) => ({ id: Date.now()+Math.random(), role, text, recipes, sub, time: ts() });

  const send = useCallback(async (txt) => {
    const text = (txt||input).trim();
    if (!text || loading) return;
    setInput(""); setLoading(true);
    setMsgs(p => [...p, addMsg("user", text)]);
    try {
      const res = await fetch(`${API}/chat`, { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({ session_id:sid, message:text }) });
      const d = await res.json();
      setPrefs(d.preferences || prefs);
      setMsgs(p => [...p, addMsg("bot", d.message, d.recipes||[], d.sub)]);
    } catch {
      setMsgs(p => [...p, addMsg("bot", "⚠️ Backend offline. Run `python backend.py` on port 8000.")]);
    } finally { setLoading(false); taRef.current?.focus(); }
  }, [input, loading, sid, prefs]);

  const sendImg = async e => {
    const file = e.target.files[0]; if (!file) return;
    setLoading(true);
    setMsgs(p => [...p, addMsg("user", `📸 ${file.name}`)]);
    const form = new FormData(); form.append("file", file); form.append("session_id", sid);
    try {
      const res = await fetch(`${API}/ocr`, { method:"POST", body:form });
      const d = await res.json();
      setMsgs(p => [...p, addMsg("bot", d.message, d.recipes||[])]);
    } catch { setMsgs(p => [...p, addMsg("bot", "⚠️ OCR failed. Please type ingredients instead.")]); }
    finally { setLoading(false); e.target.value=""; }
  };

  const resetSession = async () => {
    await fetch(`${API}/reset`, { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({ session_id:sid }) }).catch(()=>{});
    setPrefs({ exclude:[], diet:[] });
    setMsgs([addMsg("bot", "🔄 Session reset! What are we cooking today?")]);
  };

  const removeExclusion = (item) => send(`include ${item}`);

  const sc = status==="online"?C.green:status==="checking"?C.gold:C.red;
  const showQuick = msgs.length === 1;

  return <>
    <style>{`
      *{box-sizing:border-box;margin:0;padding:0}
      body{background:${C.bg};font-family:'Segoe UI',system-ui,sans-serif;color:${C.t1};height:100vh;overflow:hidden}
      ::-webkit-scrollbar{width:4px}::-webkit-scrollbar-track{background:${C.bg}}::-webkit-scrollbar-thumb{background:${C.border};border-radius:10px}
      @keyframes spin{to{transform:rotate(360deg)}}
      @keyframes up{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
      @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
      .fa{animation:up .22s ease both}
      .qb:hover{background:${C.goldG}!important;border-color:${C.gold}!important;color:${C.gold}!important}
      .excl-pill:hover{opacity:0.8;transform:scale(1.05)}
      textarea:focus{outline:none}
    `}</style>
    <div style={{display:"flex",flexDirection:"column",height:"100vh",maxWidth:860,margin:"0 auto"}}>

      {/* Header */}
      <div style={{padding:"12px 16px",background:C.surf,borderBottom:`1px solid ${C.border}`,display:"flex",alignItems:"center",justifyContent:"space-between",flexShrink:0,gap:8,flexWrap:"wrap"}}>
        <div style={{display:"flex",alignItems:"center",gap:10}}>
          <div style={{width:38,height:38,borderRadius:10,background:`linear-gradient(135deg,${C.gold},${C.goldD})`,display:"flex",alignItems:"center",justifyContent:"center",fontSize:19,boxShadow:`0 0 14px ${C.goldG}`}}>🍳</div>
          <div>
            <div style={{fontSize:15,fontWeight:700,color:C.t1}}>Recipe Assistant <span style={{fontSize:11,color:C.t3,fontWeight:400}}>v5</span></div>
            <div style={{fontSize:10,color:C.t3}}>Groq LLaMA-3 · ChromaDB · LLM Re-ranking · AI Generation</div>
          </div>
        </div>
        <div style={{display:"flex",alignItems:"center",gap:6,flexWrap:"wrap"}}>
          {prefs.exclude.map(e => (
            <span key={e} className="excl-pill" onClick={() => removeExclusion(e)} title="Click to remove this exclusion"
              style={{background:"#2a1515",border:`1px solid ${C.red}44`,color:C.red,borderRadius:20,padding:"2px 8px",fontSize:10,fontWeight:700,cursor:"pointer",transition:"all .15s"}}>
              🚫 {e} ×
            </span>
          ))}
          {prefs.diet.map(d => (
            <span key={d} style={{background:"#172140",border:`1px solid ${C.blue}44`,color:C.blue,borderRadius:20,padding:"2px 8px",fontSize:10,fontWeight:700}}>🥗 {d}</span>
          ))}
          <div style={{display:"flex",alignItems:"center",gap:5}}>
            <span style={{width:7,height:7,borderRadius:"50%",background:sc,boxShadow:`0 0 6px ${sc}`,display:"inline-block",animation:status==="checking"?"pulse 1.2s infinite":"none"}}/>
            <span style={{fontSize:11,color:C.t3}}>{status==="online"?"Online":status==="checking"?"…":"Offline"}</span>
          </div>
          <button onClick={resetSession} style={{background:C.alt,border:`1px solid ${C.border}`,borderRadius:8,color:C.t3,fontSize:11,padding:"5px 10px",cursor:"pointer"}}>↺ Reset</button>
        </div>
      </div>

      {/* Chat area */}
      <div style={{flex:1,overflowY:"auto",padding:"16px 12px 6px"}}>
        {msgs.map(msg => (
          <div key={msg.id} className="fa" style={{display:"flex",justifyContent:msg.role==="user"?"flex-end":"flex-start",marginBottom:14,gap:8,alignItems:"flex-start"}}>
            {msg.role==="bot" && <div style={{width:30,height:30,borderRadius:"50%",background:`linear-gradient(135deg,${C.gold},${C.goldD})`,display:"flex",alignItems:"center",justifyContent:"center",fontSize:14,flexShrink:0}}>🍳</div>}
            <div style={{maxWidth:"85%"}}>
              <div style={{background:msg.role==="user"?`linear-gradient(135deg,${C.gold},${C.goldD})`:C.surf,border:msg.role==="user"?"none":`1px solid ${C.border}`,borderRadius:msg.role==="user"?"16px 16px 4px 16px":"16px 16px 16px 4px",padding:"10px 14px",fontSize:13,lineHeight:1.7,color:msg.role==="user"?"#000":C.t1,fontWeight:msg.role==="user"?600:400}}>
                {msg.role==="user" ? msg.text : md(msg.text)}
              </div>
              {msg.recipes?.length > 0 && <div style={{marginTop:6}}>{msg.recipes.map((r,i) => <RecipeCard key={r.id+i} r={r} first={i===0}/>)}</div>}
              {msg.sub && <SubCard sub={msg.sub}/>}
              <span style={{fontSize:10,color:C.t3,display:"block",marginTop:3,textAlign:msg.role==="user"?"right":"left"}}>{msg.time}</span>
            </div>
            {msg.role==="user" && <div style={{width:30,height:30,borderRadius:"50%",background:C.alt,border:`1px solid ${C.border}`,display:"flex",alignItems:"center",justifyContent:"center",fontSize:12,flexShrink:0}}>👤</div>}
          </div>
        ))}

        {loading && <div style={{display:"flex",gap:8,alignItems:"center",marginBottom:14}}>
          <div style={{width:30,height:30,borderRadius:"50%",background:`linear-gradient(135deg,${C.gold},${C.goldD})`,display:"flex",alignItems:"center",justifyContent:"center",fontSize:14}}>🍳</div>
          <div style={{background:C.surf,border:`1px solid ${C.border}`,borderRadius:"16px 16px 16px 4px",padding:"10px 14px",display:"flex",gap:8,alignItems:"center"}}>
            <span style={{width:15,height:15,border:`2px solid ${C.border}`,borderTopColor:C.gold,borderRadius:"50%",display:"inline-block",animation:"spin .7s linear infinite"}}/>
            <span style={{fontSize:12,color:C.t3}}>Searching & re-ranking recipes…</span>
          </div>
        </div>}

        {showQuick && <div style={{paddingLeft:38,marginBottom:16}}>
          <p style={{fontSize:11,color:C.t3,marginBottom:8}}>💡 Try asking:</p>
          <div style={{display:"flex",flexWrap:"wrap",gap:6}}>
            {QUICK.map(q => <button key={q} className="qb" onClick={() => send(q)} style={{background:C.alt,border:`1px solid ${C.border}`,borderRadius:20,color:C.t2,fontSize:12,padding:"5px 12px",cursor:"pointer",transition:"all .15s"}}>{q}</button>)}
          </div>
        </div>}
        <div ref={endRef}/>
      </div>

      {/* Input bar */}
      <div style={{padding:"10px 12px 14px",background:C.surf,borderTop:`1px solid ${C.border}`,flexShrink:0}}>
        <div style={{display:"flex",gap:7,alignItems:"flex-end",background:C.alt,border:`1px solid ${C.border}`,borderRadius:C.r,padding:"6px 8px 6px 12px"}}>
          <button onClick={() => fileRef.current?.click()} title="Upload ingredient photo" style={{width:32,height:32,borderRadius:9,border:"none",background:C.surf,color:C.t3,fontSize:16,cursor:"pointer",display:"flex",alignItems:"center",justifyContent:"center",flexShrink:0}}>📷</button>
          <input type="file" ref={fileRef} accept="image/*" onChange={sendImg} style={{display:"none"}}/>
          <textarea ref={taRef} value={input} onChange={e => setInput(e.target.value)}
            onKeyDown={e => { if (e.key==="Enter"&&!e.shiftKey) { e.preventDefault(); send(); } }}
            onInput={e => { e.target.style.height="auto"; e.target.style.height=Math.min(e.target.scrollHeight,110)+"px"; }}
            placeholder="Type ingredients, ask for a recipe, request substitutes, or say 'random recipe'…" rows={1}
            style={{flex:1,background:"transparent",border:"none",color:C.t1,fontSize:13,lineHeight:1.5,fontFamily:"inherit",padding:"6px 0",maxHeight:110,overflowY:"auto",resize:"none"}}/>
          <button onClick={() => send()} disabled={!input.trim()||loading}
            style={{width:34,height:34,borderRadius:9,border:"none",background:input.trim()&&!loading?C.gold:"#2a2a2a",color:input.trim()&&!loading?"#000":C.t3,fontSize:16,cursor:input.trim()&&!loading?"pointer":"not-allowed",display:"flex",alignItems:"center",justifyContent:"center",flexShrink:0,transition:"all .15s"}}>➤</button>
        </div>
        <p style={{textAlign:"center",fontSize:10,color:C.t3,marginTop:5}}>Enter to send · Shift+Enter for new line · Click 🚫 pills to remove exclusions · 📷 to scan ingredients</p>
      </div>
    </div>
  </>;
}
