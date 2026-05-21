'use strict';

// ─── Constants ───────────────────────────────────────────
const PALETTE = ['#6366f1','#f59e0b','#10b981','#ef4444','#3b82f6','#ec4899','#8b5cf6','#06b6d4'];
const PIPELINE_COLORS = {
  chat_pipeline:         { bg: 'bg-purple-100', text: 'text-purple-700', border: 'border-purple-200', label: 'Chat Pipeline' },
  policy_brief_pipeline: { bg: 'bg-teal-100',   text: 'text-teal-700',   border: 'border-teal-200', label: 'Policy Brief' },
  short_chat:            { bg: 'bg-orange-100', text: 'text-orange-700', border: 'border-orange-200', label: 'Short Chat' },
};

// ─── Shared Utilities ─────────────────────────────────────
function apiBase() { return window.location.origin; }

function toNumeric(v) {
    if (typeof v === 'number') return v;
    if (typeof v !== 'string') return NaN;
    return parseFloat(v.replace(/,/g, ''));
}

function markdownToHtml(md) {
    if (!md) return '';
    try {
        return marked.parse(md);
    } catch (e) {
        console.error('Markdown error:', e);
        return `<pre class="whitespace-pre-wrap">${md}</pre>`;
    }
}

function fmtSeconds(s) {
    if (s === undefined || s === null) return '-';
    return Number(s).toFixed(1) + 's';
}

function showStatus(el, msg, type) {
    if (!el) return;
    el.textContent = msg;
    el.className = 'text-xs px-2 py-0.5 rounded border';
    if (type === 'success') el.classList.add('bg-green-100', 'text-green-800', 'border-green-200');
    else if (type === 'error') el.classList.add('bg-red-100', 'text-red-800', 'border-red-200');
    else if (type === 'warning') el.classList.add('bg-yellow-100', 'text-yellow-800', 'border-yellow-200');
    else el.classList.add('bg-gray-100', 'text-gray-600', 'border-gray-200');
}

function escapeHtml(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');
}

function renderCitations(citations) {
    if (!citations || citations.length === 0) return '';
    
    let html = `
    <details class="mt-4 border border-gray-200 rounded-lg bg-gray-50 overflow-hidden">
      <summary class="px-4 py-2 text-xs font-semibold text-gray-600 cursor-pointer hover:bg-gray-100 flex justify-between items-center select-none">
         <span>📚 แหล่งอ้างอิง (${citations.length})</span>
         <span class="text-[10px] text-gray-400">คลิกเพื่อขยาย</span>
      </summary>
      <div class="px-4 py-3 space-y-3 bg-white border-t border-gray-200">`;
    
    citations.forEach(c => {
        const code = c.citation_code || c.code || 'C-???';
        const bib = c.bibliography_text || c.reference || '';
        const url = c.open_url || c.pdf_url || '';
        const trust = c.trust_level || 'medium';
        
        let trustBadge = '';
        if (trust === 'high') trustBadge = '<span class="text-[10px] bg-green-100 text-green-700 px-1.5 py-0.5 rounded mr-1 uppercase">High</span>';
        else if (trust === 'medium') trustBadge = '<span class="text-[10px] bg-yellow-100 text-yellow-700 px-1.5 py-0.5 rounded mr-1 uppercase">Med</span>';
        
        html += `
        <div class="flex items-start group">
          <span class="shrink-0 font-mono text-[10px] bg-indigo-50 text-indigo-700 px-1.5 py-0.5 rounded border border-indigo-100 mr-2 mt-0.5">${code}</span>
          <div class="flex-1 min-w-0">
             <div class="text-xs text-gray-700 leading-relaxed">${bib}</div>
             ${url ? `<a href="${url}" target="_blank" class="text-[10px] text-indigo-600 hover:underline mt-1 inline-flex items-center">
               <svg class="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"></path></svg>
               Open Source
             </a>` : ''}
          </div>
        </div>`;
    });
    
    html += `</div></details>`;
    return html;
}

function buildChart(canvasId, spec) {
    if (!spec || !spec.data) return null;
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // Set default colors if missing
    if (spec.data.datasets) {
        spec.data.datasets.forEach((ds, i) => {
            if (!ds.borderColor && !ds.backgroundColor) {
                const color = PALETTE[i % PALETTE.length];
                ds.borderColor = color;
                ds.backgroundColor = color + '33'; // 20% opacity
                ds.borderWidth = 2;
                ds.tension = 0.3;
            }
        });
    }
    
    return new Chart(ctx, {
        type: spec.type || 'line',
        data: spec.data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: { display: true, text: spec.title || '', font: { family: 'IBM Plex Sans Thai' } },
                legend: { labels: { font: { family: 'IBM Plex Sans Thai', size: 11 } } }
            },
            scales: spec.type === 'pie' || spec.type === 'doughnut' ? {} : {
                y: { beginAtZero: true, ticks: { font: { family: 'IBM Plex Sans Thai' } } },
                x: { ticks: { font: { family: 'IBM Plex Sans Thai' } } }
            },
            ...spec.options
        }
    });
}

// ─── Tab Navigation ───────────────────────────────────────
function switchTab(name) {
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.add('hidden'));
    document.querySelectorAll('.tab-btn').forEach(b => {
        b.classList.remove('border-indigo-500', 'text-indigo-600', 'active');
        b.classList.add('border-transparent', 'text-gray-500');
    });
    
    const panel = document.getElementById(`panel-${name}`);
    const btn = document.querySelector(`.tab-btn[data-tab="${name}"]`);
    
    if (panel && btn) {
        panel.classList.remove('hidden');
        btn.classList.add('border-indigo-500', 'text-indigo-600', 'active');
        btn.classList.remove('border-transparent', 'text-gray-500');
    }
    
    if (name === 'health') TabHealth.load();
}

// ─── Health Tab ───────────────────────────────────────────
const TabHealth = {
  fetched: false,
  async load() {
    const grid = document.getElementById('health-grid');
    const badge = document.getElementById('health-badge');
    const time = document.getElementById('health-elapsed');
    const lastUpdate = document.getElementById('health-last-updated');
    
    const start = Date.now();
    grid.innerHTML = '<div class="col-span-full text-center py-10 text-gray-400">Loading system status...</div>';
    
    try {
        const res = await fetch(`${apiBase()}/api/health`);
        const data = await res.json();
        const elapsed = (Date.now() - start) / 1000;
        
        time.textContent = `Response time: ${elapsed.toFixed(2)}s`;
        lastUpdate.textContent = `Last checked: ${new Date().toLocaleTimeString()}`;
        
        this.render(data);
        
        // Update header badge
        if (data.status === 'ok' || data.status === 'healthy') {
            badge.className = 'px-2.5 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800 border border-green-200';
            badge.textContent = 'System Healthy';
        } else {
            badge.className = 'px-2.5 py-1 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800 border border-yellow-200';
            badge.textContent = 'System Degraded';
        }
    } catch (e) {
        grid.innerHTML = `<div class="col-span-full text-red-500 p-4 border border-red-200 bg-red-50 rounded-lg">Error connecting to health API: ${e.message}</div>`;
        badge.className = 'px-2.5 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800 border border-red-200';
        badge.textContent = 'Offline';
    }
  },
  
  render(data) {
    const grid = document.getElementById('health-grid');
    let html = '';
    
    const fields = [
        { key: 'status', label: 'Overall Status' },
        { key: 'version', label: 'API Version' },
        { key: 'postgresql', label: 'PostgreSQL' },
        { key: 'minio', label: 'MinIO Storage' },
        { key: 'pgvector', label: 'pgvector RAG' },
        { key: 'thaijo', label: 'ThaiJO Service' }
    ];
    
    fields.forEach(f => {
        const val = data[f.key] || data.services?.[f.key] || 'N/A';
        let statusClass = 'bg-gray-100 text-gray-800 border-gray-200';
        let statusLabel = String(val);
        
        if (val === true || val === 'ok' || val === 'healthy') {
            statusClass = 'bg-green-100 text-green-800 border-green-200';
            statusLabel = 'Connected';
        } else if (val === false || val === 'error' || val === 'failed') {
            statusClass = 'bg-red-100 text-red-800 border-red-200';
            statusLabel = 'Error';
        }
        
        html += `
        <div class="bg-white p-5 rounded-xl border border-gray-200 shadow-sm flex justify-between items-center">
           <span class="text-sm font-semibold text-gray-600">${f.label}</span>
           <span class="px-3 py-1 rounded-full text-xs font-bold border ${statusClass}">${statusLabel}</span>
        </div>`;
    });
    
    grid.innerHTML = html;
  }
};

// ─── Router Tab ───────────────────────────────────────────
const TabRouter = {
  async testRouting() {
      const input = document.getElementById('router-input');
      const message = input.value.trim();
      if (!message) return;
      
      const loading = document.getElementById('router-loading');
      const resultDiv = document.getElementById('router-result');
      
      loading.classList.remove('hidden');
      resultDiv.classList.add('hidden');
      
      try {
          const res = await fetch(`${apiBase()}/api/chat/unified`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ message: message })
          });
          
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const data = await res.json();
          this.renderResult(data);
      } catch (e) {
          resultDiv.innerHTML = `<div class="p-6 bg-red-50 text-red-700 text-sm">Error: ${e.message}</div>`;
          resultDiv.classList.remove('hidden');
      } finally {
          loading.classList.add('hidden');
      }
  },
  
  renderResult(data) {
      const resultDiv = document.getElementById('router-result');
      const routing = data.routing || {};
      const pipeline = routing.pipeline || data.pipeline_used || 'unknown';
      const colors = PIPELINE_COLORS[pipeline] || { bg: 'bg-gray-100', text: 'text-gray-700', border: 'border-gray-200', label: 'Unknown' };
      const conf = (routing.confidence || 0) * 100;
      
      resultDiv.className = `border ${colors.border} rounded-xl overflow-hidden block`;
      resultDiv.innerHTML = `
      <div class="${colors.bg} px-6 py-4 border-b ${colors.border}">
         <div class="flex justify-between items-center">
            <span class="text-sm font-bold ${colors.text} uppercase tracking-wider">Target Pipeline</span>
            <span class="px-3 py-1 rounded-full text-xs font-bold bg-white border ${colors.border} ${colors.text}">${colors.label}</span>
         </div>
      </div>
      <div class="p-6 bg-white space-y-6">
         <div>
            <label class="block text-[10px] font-bold text-gray-400 uppercase mb-2">Confidence Score</label>
            <div class="flex items-center space-x-4">
               <div class="flex-1 bg-gray-100 rounded-full h-3 overflow-hidden">
                  <div class="bg-indigo-500 h-full transition-all duration-1000" style="width: ${conf}%"></div>
               </div>
               <span class="text-sm font-bold text-gray-700">${conf.toFixed(0)}%</span>
            </div>
         </div>
         
         <div>
            <label class="block text-[10px] font-bold text-gray-400 uppercase mb-1">Decision Reason</label>
            <p class="text-sm text-gray-700 leading-relaxed">${escapeHtml(routing.reason || 'No reason provided.')}</p>
         </div>
         
         <div>
            <label class="block text-[10px] font-bold text-gray-400 uppercase mb-2">Extracted Parameters</label>
            <pre class="bg-gray-900 text-green-400 p-4 rounded-lg text-[10px] font-mono overflow-x-auto">${JSON.stringify(routing.extracted_params || {}, null, 2)}</pre>
         </div>
      </div>`;
      resultDiv.classList.remove('hidden');
  }
};

// ─── Short Chat Tab ───────────────────────────────────────
const TabShort = {
  async send() {
      const input = document.getElementById('short-input');
      const message = input.value.trim();
      if (!message) return;
      
      input.value = '';
      this.appendBubble('user', escapeHtml(message));
      
      const loading = document.getElementById('short-loading');
      loading.classList.remove('hidden');
      
      try {
          const res = await fetch(`${apiBase()}/api/chat/short`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ message: message })
          });
          
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const data = await res.json();
          
          const html = markdownToHtml(data.content);
          const meta = {
              elapsed: data.elapsed_seconds,
              citations: data.citations?.length || 0,
              pipeline: 'short_chat'
          };
          this.appendBubble('assistant', html, meta);
      } catch (e) {
          this.appendBubble('assistant', `<div class="text-red-500">Error: ${e.message}</div>`);
      } finally {
          loading.classList.add('hidden');
      }
  },
  
  appendBubble(role, html, meta) {
      const history = document.getElementById('short-chat-history');
      const bubble = document.createElement('div');
      bubble.className = `flex ${role === 'user' ? 'justify-end' : 'justify-start'}`;
      
      let metaHtml = '';
      if (meta) {
          metaHtml = `
          <div class="mt-3 pt-2 border-t border-gray-100 flex justify-between items-center text-[10px] text-gray-400">
             <span class="px-2 py-0.5 rounded bg-orange-100 text-orange-700 font-bold uppercase">Short Chat</span>
             <span>Time: ${fmtSeconds(meta.elapsed)} • ${meta.citations} Sources</span>
          </div>`;
      }
      
      bubble.innerHTML = `
      <div class="bubble ${role === 'user' ? 'bubble-user' : 'bubble-assistant prose'} shadow-sm">
         ${html}
         ${metaHtml}
      </div>`;
      
      history.appendChild(bubble);
      history.scrollTop = history.scrollHeight;
  }
};

// ─── Unified Chat Tab ─────────────────────────────────────
const TabUnified = {
  chartInstances: {},
  
  quickQuestion(text) {
      document.getElementById('unified-input').value = text;
      this.send();
  },

  async send() {
      const input = document.getElementById('unified-input');
      const message = input.value.trim();
      if (!message) return;
      
      input.value = '';
      this.appendUserBubble(message);
      
      const loading = document.getElementById('unified-loading');
      loading.classList.remove('hidden');
      
      try {
          const res = await fetch(`${apiBase()}/api/chat/unified`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ message: message })
          });
          
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const data = await res.json();
          this.appendAssistantBubble(data);
      } catch (e) {
          this.appendErrorBubble(e.message);
      } finally {
          loading.classList.add('hidden');
      }
  },

  appendUserBubble(message) {
      const history = document.getElementById('unified-chat-history');
      const bubble = document.createElement('div');
      bubble.className = 'flex justify-end';
      bubble.innerHTML = `<div class="bubble bubble-user shadow-sm">${escapeHtml(message)}</div>`;
      history.appendChild(bubble);
      history.scrollTop = history.scrollHeight;
  },

  appendAssistantBubble(data) {
      const history = document.getElementById('unified-chat-history');
      const result = data.result || {};
      const routing = data.routing || {};
      const pipeline = data.pipeline_used || 'chat_pipeline';
      const colors = PIPELINE_COLORS[pipeline] || PIPELINE_COLORS.chat_pipeline;
      
      const bubble = document.createElement('div');
      bubble.className = 'flex flex-col items-start space-y-2';
      
      const content = result.content || result.answer || result.report || result.policy_brief || '';
      const html = markdownToHtml(content);
      const charts = result.charts || (result.chart_spec ? [result.chart_spec] : []);
      const citations = result.citations || [];
      
      bubble.innerHTML = `
      <div class="bubble bubble-assistant prose shadow-sm relative overflow-hidden">
         <div class="absolute top-0 left-0 w-full h-1 ${colors.bg}">
            <div class="${colors.text.replace('text', 'bg')} h-full" style="width: ${routing.confidence * 100}%"></div>
         </div>
         <div class="flex justify-between items-center mb-4 pt-2">
            <span class="px-2 py-0.5 rounded text-[10px] font-bold ${colors.bg} ${colors.text} uppercase tracking-wider">${colors.label}</span>
            <span class="text-[10px] text-gray-400">Confidence: ${(routing.confidence * 100).toFixed(0)}%</span>
         </div>
         <div class="markdown-body">
            ${html}
         </div>
         <div class="charts-container space-y-4 mt-4"></div>
         ${renderCitations(citations)}
         <div class="mt-4 pt-3 border-t border-gray-100 flex justify-between items-center text-[10px] text-gray-400">
            <span>Pipeline: ${pipeline}</span>
            <span>Time: ${fmtSeconds(result.elapsed_seconds || result.metadata?.elapsed_seconds)}</span>
         </div>
      </div>
      <div class="ml-4 flex items-center space-x-2 text-[10px] text-gray-400 italic">
         <span>Decision: ${escapeHtml(routing.reason || 'Auto-detected')}</span>
      </div>`;
      
      history.appendChild(bubble);
      
      // Render Charts
      if (charts.length > 0) {
          const chartBox = bubble.querySelector('.charts-container');
          charts.forEach((spec, idx) => {
              const canvasId = `chart-${Date.now()}-${idx}`;
              const card = document.createElement('div');
              card.className = 'chart-card';
              card.innerHTML = `<canvas id="${canvasId}"></canvas>`;
              chartBox.appendChild(card);
              setTimeout(() => buildChart(canvasId, spec), 100);
          });
      }
      
      history.scrollTop = history.scrollHeight;
  },

  appendErrorBubble(msg) {
      const history = document.getElementById('unified-chat-history');
      const bubble = document.createElement('div');
      bubble.className = 'flex justify-start';
      bubble.innerHTML = `<div class="bubble bubble-assistant border-red-200 bg-red-50 text-red-700 text-sm">Error: ${msg}</div>`;
      history.appendChild(bubble);
      history.scrollTop = history.scrollHeight;
  },

  clear() {
      document.getElementById('unified-chat-history').innerHTML = '';
  }
};

// ─── SSE Stream Tab ───────────────────────────────────────
const TabSSE = {
  controller: null,
  
  async start() {
      const input = document.getElementById('sse-input');
      const message = input.value.trim();
      if (!message) return;
      
      const terminal = document.getElementById('sse-terminal');
      const finalBox = document.getElementById('sse-final-response');
      const startBtn = document.getElementById('sse-start-btn');
      const stopBtn = document.getElementById('sse-stop-btn');
      
      this.controller = new AbortController();
      terminal.innerHTML = `<div class="text-indigo-400 font-bold mb-2"># Starting unified stream for: "${escapeHtml(message)}"</div>`;
      finalBox.innerHTML = '';
      startBtn.disabled = true;
      stopBtn.disabled = false;
      
      try {
          const res = await fetch(`${apiBase()}/api/chat/stream`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ message: message }),
              signal: this.controller.signal
          });
          
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          
          const reader = res.body.getReader();
          const decoder = new TextDecoder();
          let buffer = '';
          
          while (true) {
              const { done, value } = await reader.read();
              if (done) break;
              
              buffer += decoder.decode(value, { stream: true });
              const lines = buffer.split('\n\n');
              buffer = lines.pop(); // Keep partial line
              
              for (const line of lines) {
                  if (line.startsWith('data: ')) {
                      try {
                          const event = JSON.parse(line.substring(6));
                          this.logEvent(event);
                          if (event.type === 'content') {
                              this.renderFinal(event.data);
                          }
                      } catch (e) {
                          console.error('JSON parse error:', e, line);
                      }
                  }
              }
          }
      } catch (e) {
          if (e.name === 'AbortError') {
              this.logRaw('> Stream aborted by user.', 'text-yellow-500');
          } else {
              this.logRaw(`> Error: ${e.message}`, 'text-red-500');
          }
      } finally {
          startBtn.disabled = false;
          stopBtn.disabled = true;
          this.logRaw('# Stream closed.', 'text-slate-500 mt-2');
      }
  },
  
  stop() {
      if (this.controller) this.controller.abort();
  },
  
  logEvent(ev) {
      if (ev.type === 'start') {
          this.logRaw(`[START] ${ev.message}`, 'text-cyan-400');
      } else if (ev.type === 'routing') {
          this.logRaw(`[ROUTER] Pipeline: ${ev.data.pipeline} (Conf: ${ev.data.confidence * 100}%)`, 'text-indigo-400');
          this.logRaw(`[ROUTER] Reason: ${ev.data.reason}`, 'text-indigo-300 italic');
      } else if (ev.type === 'agent_progress') {
          const d = ev.data || {};
          const status = d.status === 'done' ? '✓' : (d.status === 'error' ? '✗' : '⟳');
          const color = d.status === 'done' ? 'text-green-400' : (d.status === 'error' ? 'text-red-400' : 'text-sky-400');
          this.logRaw(`${status} [${d.agent_id || 'Agent'}] ${d.agent_name || ''} ${d.status}...`, color);
      } else if (ev.type === 'error') {
          this.logRaw(`[ERROR] ${ev.message}`, 'text-red-500 font-bold');
      } else if (ev.type === 'done') {
          this.logRaw(`[DONE] Pipeline: ${ev.pipeline}`, 'text-green-500 font-bold');
      }
  },
  
  logRaw(text, colorClass) {
      const terminal = document.getElementById('sse-terminal');
      const div = document.createElement('div');
      if (colorClass) div.className = colorClass;
      div.textContent = text;
      terminal.appendChild(div);
      terminal.scrollTop = terminal.scrollHeight;
  },
  
  renderFinal(data) {
      const finalBox = document.getElementById('sse-final-response');
      const bubble = document.createElement('div');
      bubble.className = 'flex justify-start';
      
      const html = markdownToHtml(data.content || data.answer || '');
      const citations = data.citations || [];
      
      bubble.innerHTML = `
      <div class="bubble bubble-assistant prose shadow-sm border border-indigo-100">
         <div class="text-[10px] font-bold text-indigo-500 uppercase mb-2">Final Streamed Result</div>
         ${html}
         ${renderCitations(citations)}
      </div>`;
      
      finalBox.appendChild(bubble);
      finalBox.scrollTop = finalBox.scrollHeight;
  }
};

// ─── Init ─────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
    // Expose Tab functions
    window.switchTab = switchTab;
    window.TabHealth = TabHealth;
    window.TabRouter = TabRouter;
    window.TabShort = TabShort;
    window.TabUnified = TabUnified;
    window.TabSSE = TabSSE;
    
    switchTab('unified');
    TabHealth.load();
});
