const S = { dataset: null, target: null, columnas: [], headers: [] };

function showPanel(id) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('panel-' + id).classList.add('active');
  const titles = { dataset:'Dataset', selectores:'Selectores',
    supervisado:'Clasificadores Supervisados', nosupervisado:'Clasificadores No Supervisados' };
  document.getElementById('topbar-title').textContent = titles[id] || id;
  event?.target?.closest('.nav-item')?.classList.add('active');
}

function toast(msg, type='info', title='') {
  const icons = { success:'✅', error:'❌', info:'ℹ️', warn:'⚠️' };
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.innerHTML = `<span style="font-size:16px">${icons[type]||'ℹ️'}</span>
    <div class="toast-msg">${title?`<div class="toast-title">${title}</div>`:''}${msg}</div>`;
  document.getElementById('toast-container').appendChild(el);
  setTimeout(() => el.remove(), 4500);
}

function loading(show, text='Procesando...') {
  const ov = document.getElementById('loading-overlay');
  document.getElementById('loading-text').textContent = text;
  ov.classList.toggle('show', show);
}

async function api(url, body, loadText) {
  loading(true, loadText || 'Procesando...');
  try {
    const isForm = body instanceof FormData;
    const r = await fetch(url, {
      method: 'POST', body: isForm ? body : JSON.stringify(body),
      headers: isForm ? {} : { 'Content-Type': 'application/json' }
    });
    const d = await r.json();
    if (d.error) throw new Error(d.error);
    return d;
  } finally { loading(false); }
}

// ── Cargar CSV ─────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const fi = document.getElementById('file-input');
  if (fi) fi.addEventListener('change', cargarCSV);
  const uz = document.getElementById('upload-zone');
  if (uz) {
    uz.addEventListener('dragover', e => { e.preventDefault(); uz.classList.add('drag-over'); });
    uz.addEventListener('dragleave', () => uz.classList.remove('drag-over'));
    uz.addEventListener('drop', e => {
      e.preventDefault(); uz.classList.remove('drag-over');
      const f = e.dataTransfer.files[0];
      if (f) { document.getElementById('file-input').files = e.dataTransfer.files; cargarCSV(); }
    });
  }
});

async function cargarCSV() {
  const fi  = document.getElementById('file-input');
  const sep = document.getElementById('sep-select').value;
  if (!fi.files.length) return;
  const fd = new FormData();
  fd.append('archivo', fi.files[0]);
  fd.append('separador', sep);
  try {
    const d = await api('/api/cargar', fd, 'Cargando dataset...');
    S.dataset = d; S.columnas = d.columnas; S.headers = d.headers; S.target = null;
    document.getElementById('stat-grid').innerHTML =
      `<div class="stat-box"><div class="stat-val">${d.filas}</div><div class="stat-lbl">Instancias</div></div>
       <div class="stat-box"><div class="stat-val">${d.columnas_count}</div><div class="stat-lbl">Atributos</div></div>`;
    const el = document.getElementById('col-selector-list');
    el.innerHTML = d.columnas.map(c => {
      const bc = c.icono==='✓'?'color:var(--success)':c.icono==='~'?'color:var(--warning)':'color:var(--blue-lit)';
      return `<div class="col-option" onclick="selTarget('${c.nombre}', this)">
        <span style="font-size:14px;${bc}">${c.icono}</span>
        <span class="col-name">${c.nombre}</span>
        <span class="col-info">${c.tipo}</span>
        ${c.nulos>0?`<span style="font-size:10px;color:var(--danger)">${c.nulos} nulos</span>`:''}
      </div>`;
    }).join('');
    const ths = d.headers.map(h=>`<th>${h}</th>`).join('');
    const trs = d.preview.map(row=>'<tr>'+d.headers.map(h=>`<td>${row[h]??''}</td>`).join('')+'</tr>').join('');
    document.getElementById('preview-table').innerHTML=`<table><thead><tr>${ths}</tr></thead><tbody>${trs}</tbody></table>`;
    document.getElementById('card-info').style.display='';
    document.getElementById('card-preview').style.display='';
    updateStatus('dataset', true, `${d.filas} × ${d.columnas_count}`);
    toast(`Dataset cargado: ${d.nombre}`, 'success', 'Listo');
  } catch(e) { toast(e.message, 'error', 'Error'); }
}

async function selTarget(col, el) {
  document.querySelectorAll('.col-option').forEach(e=>e.classList.remove('selected'));
  el.classList.add('selected');
  try {
    const d = await api('/api/set_target', {columna:col}, 'Configurando target...');
    S.target = col;
    updateStatus('target', true, `Target: ${col} (${d.n_clases} clases)`);
    toast(`Target: ${col} — ${d.n_clases} clases`, 'success');
  } catch(e) { toast(e.message, 'error', 'Error'); }
}

function updateStatus(key, ok, text) {
  const dot = document.getElementById(`dot-${key}`);
  const lbl = document.getElementById(`status-${key}`);
  if (dot) dot.className = 'dot ' + (ok ? 'ok' : 'warn');
  if (lbl) lbl.textContent = text;
}

// ── Selectores ─────────────────────────────────────────────────────────
async function ejecutarSelector(metodo) {
  if (!S.target) { toast('Selecciona el target primero', 'warn'); return; }
  const label = {correlacion:'Correlación',chi2:'Chi²/F-score',
    random_forest:'Random Forest',gower:'Gower',todos:'Todos'}[metodo]||metodo;
  try {
    const d = await api('/api/selector/'+metodo, {}, `Ejecutando ${label}...`);
    const card = document.getElementById('card-sel-result');
    card.style.display='';
    document.getElementById('sel-result-title').textContent=label;
    const campo = d.resultados[0]?.SCORE_FINAL!==undefined?'SCORE_FINAL':'score';
    const cols  = metodo==='todos'
      ? ['variable','Correlacion','Chi2_F','RandomForest','Gower','SCORE_FINAL']
      : ['variable','score'];
    const avail = cols.filter(c=>d.resultados[0]?.[c]!==undefined);
    let html=`<table><thead><tr>${avail.map(c=>`<th>${c}</th>`).join('')}</tr></thead><tbody>`;
    d.resultados.forEach((r,i)=>{
      const hi=i<3?'color:var(--success)':i<6?'color:var(--blue-lit)':'';
      html+=`<tr>${avail.map(c=>{const v=r[c];return`<td style="${typeof v==='number'?hi:''}">${typeof v==='number'?v.toFixed(6):v}</td>`}).join('')}</tr>`;
    });
    html+='</tbody></table>';
    document.getElementById('sel-result-table').innerHTML=html;
    if(d.grafica) document.getElementById('sel-result-graph').innerHTML=`<img src="${d.grafica}">`;
    toast(`${label} completado`, 'success');
  } catch(e) { toast(e.message, 'error', 'Error'); }
}

// ── Clasificadores ─────────────────────────────────────────────────────
function toggleCheck(el) { el.classList.toggle('checked'); }

function getChecked(cid) {
  return [...document.querySelectorAll(`#${cid} .check-item.checked`)].map(e=>e.dataset.val);
}

async function ejecutarClasificadores(tipo) {
  if (!S.target) { toast('Selecciona target primero','warn'); return; }
  const cid    = tipo==='supervisado'?'clf-sup-checks':'clf-nosup-checks';
  const modelos = getChecked(cid);
  if (!modelos.length) { toast('Selecciona al menos un clasificador','warn'); return; }
  const test_size = parseFloat(document.getElementById(`${tipo}-test-size`).value)/100;
  try {
    const d = await api('/api/clasificar',
      {tipo, modelos, test_size, params:{}}, 'Entrenando...');
    const wrap = document.getElementById(`${tipo}-results`);
    wrap.style.display='';
    let html=`<table><thead><tr><th>Modelo</th><th>Accuracy</th><th>F1</th><th>CV-5</th><th>±std</th></tr></thead><tbody>`;
    const mejor=d.resultados.reduce((a,b)=>a.cv_mean>b.cv_mean?a:b);
    d.resultados.forEach(r=>{
      const isBest=r.modelo===mejor.modelo;
      html+=`<tr><td style="font-weight:700;color:${isBest?'var(--success)':'var(--white)'}">${isBest?'⭐ ':''}${r.modelo}</td>
        <td style="color:var(--blue-lit)">${r.accuracy.toFixed(6)}</td>
        <td style="color:var(--teal)">${r.f1.toFixed(6)}</td>
        <td style="color:var(--success)">${r.cv_mean.toFixed(6)}</td>
        <td style="color:var(--muted)">±${r.cv_std.toFixed(6)}</td></tr>`;
    });
    html+='</tbody></table>';
    document.getElementById(`${tipo}-table`).innerHTML=html;
    if(d.grafica) document.getElementById(`${tipo}-graph`).innerHTML=`<img src="${d.grafica}">`;
    toast(`${tipo} completado — ${modelos.length} modelo(s)`, 'success');
  } catch(e) { toast(e.message,'error','Error'); }
}