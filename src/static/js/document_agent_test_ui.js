'use strict';
const API = window.location.origin;

// ─── Shared Utilities ──────────────────────────────────────────
function escapeHtml(s) {
    if (!s) return '';
    return String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}
function fmtBytes(n) {
    if (n === 0 || !n) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(n) / Math.log(k));
    return parseFloat((n / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}
function fmtDate(s) {
    if (!s) return '-';
    const d = new Date(s);
    return d.toLocaleDateString('th-TH', { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}
function statusBadge(s) {
    if (!s) return '';
    const lower = s.toLowerCase();
    let color = 'gray';
    if (lower === 'completed') color = 'green';
    else if (lower === 'failed') color = 'red';
    else if (lower === 'pending') color = 'yellow';
    else if (lower === 'processing') color = 'blue';
    return `<span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-${color}-100 text-${color}-800">${s}</span>`;
}
function topicBadge(t) {
    if (!t) return '';
    const lower = t.toLowerCase();
    let color = 'gray';
    let icon = '';
    if (lower === 'accident') { color = 'red'; icon = '🚗 '; }
    else if (lower === 'mental_health') { color = 'amber'; icon = '🧠 '; }
    else if (lower === 'nutrition') { color = 'emerald'; icon = '🍎 '; }
    return `<span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-${color}-50 text-${color}-700 border border-${color}-200">${icon}${t}</span>`;
}
function trustedBadge(t) {
    if (!t) return '';
    const lower = t.toLowerCase();
    let color = 'gray';
    if (lower === 'high') color = 'green';
    else if (lower === 'medium') color = 'yellow';
    else if (lower === 'low') color = 'red';
    return `<span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-${color}-100 text-${color}-800 border border-${color}-200 uppercase">${lower}</span>`;
}

// ─── Tab Navigation ────────────────────────────────────────────
const TABS = ['rag','indicators','sql','library','upload','evidence'];
function switchTab(name) {
    TABS.forEach(tab => {
        const panel = document.getElementById(`panel-${tab}`);
        const btn = document.querySelector(`.tab-btn[data-tab="${tab}"]`);
        if (panel && btn) {
            if (tab === name) {
                panel.classList.remove('hidden');
                btn.classList.add('border-indigo-500', 'text-indigo-600');
                btn.classList.remove('border-transparent', 'text-gray-500');
            } else {
                panel.classList.add('hidden');
                btn.classList.remove('border-indigo-500', 'text-indigo-600');
                btn.classList.add('border-transparent', 'text-gray-500');
            }
        }
    });
}

// ─── Health Badge ──────────────────────────────────────────────
async function checkHealth() {
    const badge = document.getElementById('health-badge');
    try {
        const res = await fetch(`${API}/api/health`);
        if (res.ok) {
            const data = await res.json();
            if (data.status === 'ok' || data.status === 'healthy') {
                 badge.className = 'px-2.5 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800 border border-green-200';
                 badge.textContent = 'System Healthy';
            } else {
                 badge.className = 'px-2.5 py-1 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800 border border-yellow-200';
                 badge.textContent = 'System Degraded';
            }
        } else {
            badge.className = 'px-2.5 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800 border border-red-200';
            badge.textContent = 'System Offline';
        }
    } catch (e) {
        badge.className = 'px-2.5 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800 border border-red-200';
        badge.textContent = 'Connection Error';
    }
}

// ─── Tab: RAG Search ──────────────────────────────────────────
const TabRAG = {
  async search() {
      const keywords = document.getElementById('rag-keywords').value;
      const topic = document.getElementById('rag-topic').value;
      const n = document.getElementById('rag-n').value;
      
      const loading = document.getElementById('rag-loading');
      const resultsContainer = document.getElementById('rag-results');
      
      if (!keywords) {
          alert('Please enter keywords to search.');
          return;
      }
      
      loading.classList.remove('hidden');
      resultsContainer.innerHTML = '';
      
      try {
          const res = await fetch(`${API}/api/test/tool/search_documents`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ topic: topic, keywords: keywords, n_results: parseInt(n) })
          });
          
          if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
          
          const data = await res.json();
          // The endpoint returns JSON string in its 'result' or might return the array directly.
          // Parse it properly.
          let items = [];
          if (data && typeof data === 'string') {
              try { items = JSON.parse(data); } catch(e) {}
          } else if (data && data.result && typeof data.result === 'string') {
              try { items = JSON.parse(data.result); } catch(e) {}
          } else if (Array.isArray(data)) {
              items = data;
          } else if (data && Array.isArray(data.items)) {
              items = data.items;
          }
          
          TabRAG.renderResults(items);
      } catch (e) {
          console.error(e);
          resultsContainer.innerHTML = `<div class="p-4 bg-red-50 text-red-700 rounded border border-red-200">Error: ${escapeHtml(e.message)}</div>`;
      } finally {
          loading.classList.add('hidden');
      }
  },
  
  renderResults(items) {
      const container = document.getElementById('rag-results');
      if (!items || items.length === 0) {
          container.innerHTML = `<div class="text-center py-10 text-gray-500 border-2 border-dashed border-gray-200 rounded-lg">ไม่พบเอกสาร — ลองเปลี่ยน keywords</div>`;
          return;
      }
      
      let html = '';
      items.forEach((item, index) => {
          const relevance = TabRAG.relevanceBar(item.distance || 0);
          const title = item.title || item.source_ref || 'Unknown Document';
          const snippet = item.text_snippet ? escapeHtml(item.text_snippet.substring(0, 300)) + (item.text_snippet.length > 300 ? '...' : '') : '';
          const bibText = item.bibliography_text || item.apa_citation || '';
          
          let pdfBtn = '';
          if (item.document_id) {
              pdfBtn = `<a href="${API}/api/documents/open/${item.document_id}" target="_blank" class="text-xs bg-indigo-50 hover:bg-indigo-100 text-indigo-700 border border-indigo-200 px-2 py-1 rounded transition-colors inline-flex items-center"><span class="mr-1">📄</span> [PDF]</a>`;
          }
          
          html += `
          <div class="bg-white border border-gray-200 rounded-lg p-4 shadow-sm hover:shadow-md transition-shadow">
             <div class="flex justify-between items-start mb-2">
                <h3 class="font-semibold text-gray-800 text-sm truncate flex-1 pr-2" title="${escapeHtml(title)}">${escapeHtml(title)}</h3>
                <div class="flex space-x-1 shrink-0">
                    ${item.document_id ? `<span class="bg-gray-100 text-gray-600 text-xs px-2 py-0.5 rounded border border-gray-200">ID: ${item.document_id}</span>` : ''}
                    ${topicBadge(item.topic)}
                </div>
             </div>
             
             <div class="mb-3">
                 <div class="flex justify-between text-xs text-gray-500 mb-1">
                     <span>Relevance (Distance: ${Number(item.distance || 0).toFixed(4)})</span>
                     <span>${((1 - (item.distance || 0)) * 100).toFixed(1)}%</span>
                 </div>
                 ${relevance}
             </div>
             
             <div class="bg-gray-50 p-3 rounded border border-gray-100 mb-3">
                <p class="text-xs font-mono text-gray-600 whitespace-pre-wrap">${snippet}</p>
             </div>
             
             <div class="flex justify-between items-center mt-3 pt-3 border-t border-gray-100">
                 <div class="text-xs text-gray-500 italic truncate flex-1 pr-4" title="${escapeHtml(bibText)}">
                    ${escapeHtml(bibText)}
                 </div>
                 <div class="flex items-center space-x-2">
                     ${trustedBadge(item.trust_level)}
                     ${pdfBtn}
                 </div>
             </div>
          </div>`;
      });
      container.innerHTML = html;
  },
  
  relevanceBar(distance) {
      // 0 = perfect match (green, 100%), 1 = poor match (red, 0%)
      // If distance > 1, cap it at 1.
      const d = Math.max(0, Math.min(1, distance));
      const percentage = (1 - d) * 100;
      
      let colorClass = 'bg-red-500';
      if (percentage >= 75) colorClass = 'bg-green-500';
      else if (percentage >= 50) colorClass = 'bg-yellow-500';
      else if (percentage >= 25) colorClass = 'bg-orange-500';
      
      return `
      <div class="w-full bg-gray-200 rounded-full h-1.5">
         <div class="${colorClass} h-1.5 rounded-full" style="width: ${percentage}%"></div>
      </div>`;
  }
};

// ─── Tab: Indicator Catalog ───────────────────────────────────
const TabIndicators = {
  async load(topic) {
      // Update active button state
      document.querySelectorAll('.topic-btn-ind').forEach(btn => {
          btn.classList.remove('active', 'bg-teal-100', 'text-teal-800', 'border-teal-300');
          btn.classList.add('bg-gray-100', 'text-gray-700', 'border-gray-200');
      });
      const activeBtn = document.querySelector(`.topic-btn-ind[onclick*="'${topic}'"]`);
      if (activeBtn) {
          activeBtn.classList.remove('bg-gray-100', 'text-gray-700', 'border-gray-200');
          activeBtn.classList.add('active', 'bg-teal-100', 'text-teal-800', 'border-teal-300');
      }

      const loading = document.getElementById('indicators-loading');
      const container = document.getElementById('indicators-container');
      
      loading.classList.remove('hidden');
      container.innerHTML = '';
      
      try {
          const res = await fetch(`${API}/api/test/tool/indicator_catalog`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ topic: topic })
          });
          
          if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
          
          const data = await res.json();
          let indicators = [];
          
          if (data && typeof data === 'string') {
              try { indicators = JSON.parse(data); } catch(e) {}
          } else if (data && data.result && typeof data.result === 'string') {
              try { indicators = JSON.parse(data.result); } catch(e) {}
          } else if (Array.isArray(data)) {
              indicators = data;
          } else if (data && Array.isArray(data.items)) {
              indicators = data.items;
          }
          
          TabIndicators.renderTable(indicators, topic);
      } catch (e) {
          console.error(e);
          container.innerHTML = `<div class="p-4 bg-red-50 text-red-700 rounded border border-red-200">Error: ${escapeHtml(e.message)}</div>`;
      } finally {
          loading.classList.add('hidden');
      }
  },
  
  renderTable(indicatorsData, topic) {
      const container = document.getElementById('indicators-container');
      if (!indicatorsData || indicatorsData.length === 0) {
          container.innerHTML = `<div class="text-center py-10 text-gray-500 border-2 border-dashed border-gray-200 rounded-lg">ไม่พบตัวชี้วัดสำหรับหัวข้อ ${escapeHtml(topic)}</div>`;
          return;
      }
      
      // Parse plain text string into array of objects if needed
      let indicators = [];
      if (typeof indicatorsData === 'string') {
          const lines = indicatorsData.split('\n');
          lines.forEach(line => {
              if (line.trim().startsWith('-')) {
                  // e.g. "- ACC-001: จำนวนอุบัติเหตุ (ครั้ง) [chart: line]"
                  // Filter out lines that look like mojibake (with '?')
                  if (!line.includes('?')) {
                      const match = line.match(/- ([A-Z0-9-]+): (.*?) \((.*?)\) \[chart: (.*?)\]/);
                      if (match) {
                          indicators.push({
                              indicator_code: match[1],
                              indicator_name: match[2],
                              unit_name: match[3],
                              preferred_chart: match[4],
                              definition: 'No definition available in this view.' // Or fetch it if possible
                          });
                      }
                  }
              }
          });
      } else {
          indicators = indicatorsData;
      }
      
      // Remove duplicates based on code due to mojibake data insertion previously
      const uniqueIndicators = [];
      const seenCodes = new Set();
      for (const ind of indicators) {
          const code = ind.indicator_code || ind.code;
          if (!seenCodes.has(code)) {
              seenCodes.add(code);
              uniqueIndicators.push(ind);
          }
      }
      indicators = uniqueIndicators;
      
      if (indicators.length === 0) {
          container.innerHTML = `<div class="text-center py-10 text-gray-500 border-2 border-dashed border-gray-200 rounded-lg">ไม่พบตัวชี้วัดสำหรับหัวข้อ ${escapeHtml(topic)}</div>`;
          return;
      }

      let html = `
      <div class="mb-4 text-sm text-gray-600 font-medium">พบ ${indicators.length} ตัวชี้วัด</div>
      <div class="overflow-x-auto border border-gray-200 rounded-lg">
        <table class="min-w-full divide-y divide-gray-200">
          <thead class="bg-gray-50">
            <tr>
              <th scope="col" class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Code</th>
              <th scope="col" class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Name</th>
              <th scope="col" class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Unit</th>
              <th scope="col" class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Chart</th>
            </tr>
          </thead>
          <tbody class="bg-white divide-y divide-gray-200">
      `;
      
      indicators.forEach((ind, i) => {
          html += `
            <tr class="hover:bg-gray-50 cursor-pointer transition-colors" onclick="document.getElementById('ind-desc-${i}').classList.toggle('hidden')">
              <td class="px-4 py-3 whitespace-nowrap text-sm font-medium text-gray-900">${escapeHtml(ind.indicator_code || ind.code)}</td>
              <td class="px-4 py-3 text-sm text-gray-700">${escapeHtml(ind.indicator_name || ind.name)}</td>
              <td class="px-4 py-3 whitespace-nowrap text-sm text-gray-500">${escapeHtml(ind.unit_name || ind.unit || '-')}</td>
              <td class="px-4 py-3 whitespace-nowrap text-sm text-gray-500">${escapeHtml(ind.preferred_chart || ind.chart || '-')}</td>
            </tr>
            <tr id="ind-desc-${i}" class="hidden bg-teal-50/30">
              <td colspan="4" class="px-4 py-3 text-sm text-gray-600 border-t border-teal-100">
                 <div class="font-medium text-teal-800 mb-1">Definition:</div>
                 <div class="pl-2 border-l-2 border-teal-300 text-xs text-gray-600 whitespace-pre-wrap">${escapeHtml(ind.definition || ind.desc || 'No definition available')}</div>
              </td>
            </tr>
          `;
      });
      
      html += `</tbody></table></div>`;
      container.innerHTML = html;
  }
};

// ─── Tab: SQL Playground ──────────────────────────────────────
const TabSQL = {
  async execute() {
      const sql = document.getElementById('sql-input').value.trim();
      if (!sql) return;
      
      // Basic client-side check
      if (!sql.toUpperCase().startsWith('SELECT') && !sql.toUpperCase().startsWith('WITH')) {
          const errDiv = document.getElementById('sql-error');
          errDiv.classList.remove('hidden');
          errDiv.textContent = 'Only SELECT or WITH queries are allowed.';
          return;
      }
      
      const loading = document.getElementById('sql-loading');
      const container = document.getElementById('sql-results-container');
      const errDiv = document.getElementById('sql-error');
      const rowCountSpan = document.getElementById('sql-row-count');
      const jsonContainer = document.getElementById('sql-json-container');
      
      loading.classList.remove('hidden');
      container.innerHTML = '';
      errDiv.classList.add('hidden');
      rowCountSpan.textContent = '';
      jsonContainer.textContent = '';
      
      try {
          const res = await fetch(`${API}/api/test/query`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ sql: sql })
          });
          
          const data = await res.json();
          
          // API format might be wrapped
          let resultData = data;
          if (data && data.result) {
               if (typeof data.result === 'string') {
                    try { resultData = JSON.parse(data.result); } catch(e) { resultData = { rows: [] }; }
               } else {
                    resultData = data.result;
               }
          }
          
          if (!res.ok || (resultData && resultData.success === false)) {
               throw new Error(resultData.error || data.detail || 'SQL execution failed');
          }
          
          const rows = resultData.rows || [];
          TabSQL.renderTable(rows);
          
          jsonContainer.textContent = JSON.stringify(rows, null, 2);
          
      } catch (e) {
          console.error(e);
          errDiv.classList.remove('hidden');
          errDiv.textContent = escapeHtml(e.message);
          container.innerHTML = `<div class="text-center py-10 text-red-400">Query failed.</div>`;
      } finally {
          loading.classList.add('hidden');
      }
  },
  
  async describeTable() {
      const tableName = document.getElementById('schema-table-select').value;
      const loading = document.getElementById('schema-loading');
      const results = document.getElementById('schema-results');
      
      loading.classList.remove('hidden');
      results.classList.add('hidden');
      
      try {
          const sql = `SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '${tableName}' ORDER BY ordinal_position;`;
          
          const res = await fetch(`${API}/api/test/query`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ sql: sql })
          });
          
          const data = await res.json();
          let resultData = data;
          if (data && data.result) {
               if (typeof data.result === 'string') {
                    try { resultData = JSON.parse(data.result); } catch(e) { resultData = { rows: [] }; }
               } else {
                    resultData = data.result;
               }
          }
          
          const rows = resultData.rows || [];
          
          if (rows.length === 0) {
              results.innerHTML = `<div class="p-3 text-gray-500 text-center">Table not found or empty schema.</div>`;
          } else {
              let html = `
              <table class="min-w-full divide-y divide-gray-200">
                <thead class="bg-gray-100">
                  <tr>
                    <th class="px-3 py-2 text-left text-xs font-semibold text-gray-600">Column</th>
                    <th class="px-3 py-2 text-left text-xs font-semibold text-gray-600">Type</th>
                  </tr>
                </thead>
                <tbody class="bg-white divide-y divide-gray-100">
              `;
              rows.forEach(r => {
                  html += `
                  <tr class="hover:bg-gray-50">
                    <td class="px-3 py-1.5 text-xs text-gray-800 font-mono">${escapeHtml(r.column_name)}</td>
                    <td class="px-3 py-1.5 text-xs text-gray-500 font-mono">${escapeHtml(r.data_type)}</td>
                  </tr>`;
              });
              html += `</tbody></table>`;
              results.innerHTML = html;
          }
          results.classList.remove('hidden');
          
      } catch (e) {
          console.error(e);
          results.innerHTML = `<div class="p-3 text-red-500 text-xs">Error: ${escapeHtml(e.message)}</div>`;
          results.classList.remove('hidden');
      } finally {
          loading.classList.add('hidden');
      }
  },
  
  renderTable(rows) {
      const container = document.getElementById('sql-results-container');
      const rowCountSpan = document.getElementById('sql-row-count');
      
      if (!rows || rows.length === 0) {
          container.innerHTML = `<div class="text-center py-10 text-gray-500">Query returned 0 rows.</div>`;
          rowCountSpan.textContent = '(0 rows)';
          return;
      }
      
      rowCountSpan.textContent = `(${rows.length} rows)`;
      
      // Extract headers from first row
      const headers = Object.keys(rows[0]);
      
      let html = `
      <table class="min-w-full divide-y divide-gray-200">
        <thead class="bg-gray-100 sticky top-0">
          <tr>
      `;
      
      headers.forEach(h => {
          html += `<th scope="col" class="px-4 py-2 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider whitespace-nowrap">${escapeHtml(h)}</th>`;
      });
      
      html += `</tr></thead><tbody class="bg-white divide-y divide-gray-200">`;
      
      rows.forEach(row => {
          html += `<tr class="hover:bg-amber-50/30">`;
          headers.forEach(h => {
              let val = row[h];
              if (val === null || val === undefined) val = 'NULL';
              else if (typeof val === 'object') val = JSON.stringify(val);
              
              // Apply simple formatting based on type/value
              let extraClass = '';
              if (val === 'NULL') extraClass = 'text-gray-400 italic';
              else if (typeof row[h] === 'number') extraClass = 'text-blue-600 font-mono text-right';
              else extraClass = 'text-gray-800';
              
              html += `<td class="px-4 py-2 text-sm whitespace-nowrap ${extraClass}">${escapeHtml(String(val))}</td>`;
          });
          html += `</tr>`;
      });
      
      html += `</tbody></table>`;
      container.innerHTML = html;
  },
  
  toggleJson() {
      const showJson = document.getElementById('sql-json-toggle').checked;
      const tableContainer = document.getElementById('sql-results-container');
      const jsonContainer = document.getElementById('sql-json-container');
      
      if (showJson) {
          tableContainer.classList.add('hidden');
          jsonContainer.classList.remove('hidden');
      } else {
          tableContainer.classList.remove('hidden');
          jsonContainer.classList.add('hidden');
      }
  }
};

// ─── Tab: Document Library ────────────────────────────────────
const TabLibrary = {
  page: 1,
  perPage: 20,
  
  async load(pageNum) {
      if (pageNum !== undefined) this.page = pageNum;
      
      const topic = document.getElementById('lib-topic').value;
      const status = document.getElementById('lib-status').value;
      const search = document.getElementById('lib-search').value;
      
      const loading = document.getElementById('lib-loading');
      const tbody = document.getElementById('lib-tbody');
      const info = document.getElementById('lib-pagination-info');
      
      loading.classList.remove('hidden');
      
      try {
          const params = new URLSearchParams({
              topic: topic,
              status: status,
              search: search,
              page: this.page,
              per_page: this.perPage
          });
          
          const res = await fetch(`${API}/api/documents?${params.toString()}`);
          if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
          
          const data = await res.json();
          this.renderList(data.documents || [], data.total || 0);
          
      } catch (e) {
          console.error(e);
          tbody.innerHTML = `<tr><td colspan="6" class="px-6 py-10 text-center text-red-500">Error: ${escapeHtml(e.message)}</td></tr>`;
      } finally {
          loading.classList.add('hidden');
      }
  },
  
  renderList(docs, total) {
      const tbody = document.getElementById('lib-tbody');
      const info = document.getElementById('lib-pagination-info');
      const btnPrev = document.getElementById('lib-btn-prev');
      const btnNext = document.getElementById('lib-btn-next');
      
      if (!docs || docs.length === 0) {
          tbody.innerHTML = `<tr><td colspan="6" class="px-6 py-10 text-center text-gray-500">ไม่พบเอกสารตามเงื่อนไขที่ระบุ</td></tr>`;
          info.textContent = `Showing 0 of 0`;
          btnPrev.disabled = true;
          btnNext.disabled = true;
          return;
      }
      
      let html = '';
      docs.forEach(doc => {
          const date = doc.uploaded_at ? fmtDate(doc.uploaded_at) : '-';
          
          html += `
          <tr class="hover:bg-gray-50 cursor-pointer transition-colors" onclick="TabLibrary.loadDetail(${doc.document_id}, this)">
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${doc.document_id}</td>
            <td class="px-6 py-4">
               <div class="text-sm font-medium text-gray-900 truncate max-w-xs" title="${escapeHtml(doc.title)}">${escapeHtml(doc.title || '-')}</div>
               <div class="flex items-center text-xs text-gray-500 mt-1 space-x-2">
                  <span title="File Type" class="uppercase">${escapeHtml(doc.document_type || '-')}</span>
                  <span>•</span>
                  <span>${fmtBytes(doc.file_size)}</span>
               </div>
            </td>
            <td class="px-6 py-4 whitespace-nowrap">${topicBadge(doc.topic)}</td>
            <td class="px-6 py-4 whitespace-nowrap">${statusBadge(doc.ingestion_status || doc.status)}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
               <div><span class="font-medium text-gray-700">${doc.chunk_count || 0}</span> chunks</div>
               <div class="text-xs">${doc.total_pages || 0} pages</div>
            </td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${date}</td>
          </tr>
          <tr id="lib-detail-${doc.document_id}" class="hidden bg-gray-50/50">
            <td colspan="6" class="p-0 border-t border-gray-100">
               <div class="p-6">
                  <div class="text-center py-4 text-purple-500 loading-dots"><span>.</span><span>.</span><span>.</span></div>
               </div>
            </td>
          </tr>
          `;
      });
      
      tbody.innerHTML = html;
      
      const start = (this.page - 1) * this.perPage + 1;
      const end = Math.min(this.page * this.perPage, total);
      info.textContent = `Showing ${start}-${end} of ${total}`;
      
      btnPrev.disabled = this.page <= 1;
      btnNext.disabled = this.page * this.perPage >= total;
  },
  
  prevPage() {
      if (this.page > 1) {
          this.page--;
          this.load();
      }
  },
  
  nextPage() {
      this.page++;
      this.load();
  },
  
  async loadDetail(id, rowEl) {
      const detailRow = document.getElementById(`lib-detail-${id}`);
      
      // Close others if opening a new one
      document.querySelectorAll('[id^="lib-detail-"]').forEach(el => {
          if (el.id !== `lib-detail-${id}`) el.classList.add('hidden');
      });
      
      if (!detailRow.classList.contains('hidden')) {
          detailRow.classList.add('hidden');
          return;
      }
      
      detailRow.classList.remove('hidden');
      
      try {
          const res = await fetch(`${API}/api/documents/${id}`);
          if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
          
          const doc = await res.json();
          TabLibrary.renderDetail(id, doc, detailRow);
          
      } catch (e) {
          console.error(e);
          detailRow.querySelector('td > div').innerHTML = `<div class="p-4 text-red-500">Error loading details: ${escapeHtml(e.message)}</div>`;
      }
  },
  
  renderDetail(id, doc, detailRow) {
      const apa = doc.apa || {};
      const chunks = doc.chunks_preview || [];
      const apaCitation = apa.formatted || doc.apa_citation || doc.bibliography_text || 'No APA citation available.';
      
      let chunkHtml = '';
      if (chunks.length === 0) {
          chunkHtml = '<div class="text-sm text-gray-500 p-3 text-center bg-white border border-gray-200 rounded">No chunks available.</div>';
      } else {
          chunkHtml = `
          <div class="border border-gray-200 rounded overflow-hidden">
             <table class="min-w-full divide-y divide-gray-200">
               <thead class="bg-gray-100">
                 <tr>
                   <th class="px-3 py-2 text-left text-xs font-semibold text-gray-600">Idx</th>
                   <th class="px-3 py-2 text-left text-xs font-semibold text-gray-600">Section</th>
                   <th class="px-3 py-2 text-left text-xs font-semibold text-gray-600">Snippet</th>
                 </tr>
               </thead>
               <tbody class="bg-white divide-y divide-gray-100 text-xs">
          `;
          chunks.forEach(c => {
              const snippet = c.text_preview ? escapeHtml(c.text_preview.substring(0, 150)) + '...' : (c.document ? escapeHtml(c.document.substring(0, 150)) + '...' : '');
              chunkHtml += `
                 <tr class="hover:bg-gray-50">
                   <td class="px-3 py-2 whitespace-nowrap text-gray-500">${c.chunk_index}</td>
                   <td class="px-3 py-2 text-gray-700 font-medium truncate max-w-[150px]" title="${escapeHtml(c.section_label)}">${escapeHtml(c.section_label || '-')}</td>
                   <td class="px-3 py-2 text-gray-600 font-mono">${snippet}</td>
                 </tr>
              `;
          });
          chunkHtml += `</tbody></table></div>`;
      }
      
      const html = `
      <div class="p-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
         <!-- Left Col: Meta & Actions -->
         <div class="space-y-4">
            <div class="bg-white p-4 rounded-lg border border-gray-200 shadow-sm">
               <h4 class="text-sm font-semibold text-gray-800 mb-3 border-b border-gray-100 pb-2">Document Actions</h4>
               <div class="flex flex-col space-y-2">
                  <a href="${API}/api/documents/open/${id}" target="_blank" class="w-full text-center bg-indigo-50 hover:bg-indigo-100 text-indigo-700 border border-indigo-200 px-4 py-2 rounded transition-colors text-sm font-medium">📄 Open Document [PDF]</a>
                  <div class="flex space-x-2 mt-2">
                     <button onclick="TabLibrary.reingest(${id})" class="flex-1 bg-white hover:bg-gray-50 text-gray-700 border border-gray-300 px-3 py-1.5 rounded transition-colors text-xs font-medium">🔄 Re-ingest</button>
                     <button onclick="TabLibrary.deleteDoc(${id}, '${escapeHtml(doc.title)}')" class="flex-1 bg-red-50 hover:bg-red-100 text-red-700 border border-red-200 px-3 py-1.5 rounded transition-colors text-xs font-medium">🗑️ Delete</button>
                  </div>
               </div>
            </div>
            
            <div class="bg-white p-4 rounded-lg border border-gray-200 shadow-sm">
               <h4 class="text-sm font-semibold text-gray-800 mb-2 border-b border-gray-100 pb-2">File Info</h4>
               <div class="space-y-1 text-xs">
                  <div class="flex justify-between"><span class="text-gray-500">Method:</span> <span class="font-medium text-gray-800">${escapeHtml(doc.upload_method || 'minio')}</span></div>
                  <div class="flex justify-between"><span class="text-gray-500">Size:</span> <span class="font-medium text-gray-800">${fmtBytes(doc.file_size)}</span></div>
                  <div class="flex justify-between"><span class="text-gray-500">Pages:</span> <span class="font-medium text-gray-800">${doc.total_pages || 0}</span></div>
                  <div class="flex justify-between"><span class="text-gray-500">Source:</span> <span class="font-medium text-gray-800">${escapeHtml(doc.source_type || 'internal')}</span></div>
                  <div class="mt-2 text-gray-400 break-all">${escapeHtml(doc.file_path || doc.minio_path || '-')}</div>
               </div>
            </div>
         </div>
         
         <!-- Mid Col: APA -->
         <div class="lg:col-span-2 space-y-4">
            <div class="bg-white p-4 rounded-lg border border-gray-200 shadow-sm">
               <div class="flex justify-between items-center mb-3 border-b border-gray-100 pb-2">
                  <h4 class="text-sm font-semibold text-gray-800">APA Citation</h4>
                  <button onclick="navigator.clipboard.writeText('${escapeHtml(apaCitation)}'); alert('Copied!')" class="text-xs bg-gray-100 hover:bg-gray-200 text-gray-600 px-2 py-1 rounded transition-colors">Copy</button>
               </div>
               <div class="p-3 bg-purple-50 text-purple-900 rounded border border-purple-100 text-sm mb-4">
                  ${escapeHtml(apaCitation)}
               </div>
               
               <div class="grid grid-cols-2 md:grid-cols-3 gap-3 text-xs">
                  <div>
                     <div class="text-gray-500 mb-0.5">APA Type</div>
                     <div class="font-medium text-gray-800">${escapeHtml(apa.type || doc.apa_type || '-')}</div>
                  </div>
                  <div>
                     <div class="text-gray-500 mb-0.5">Year</div>
                     <div class="font-medium text-gray-800">${escapeHtml(apa.year || doc.apa_year || '-')}</div>
                  </div>
                  <div>
                     <div class="text-gray-500 mb-0.5">Authors</div>
                     <div class="font-medium text-gray-800 truncate" title="${escapeHtml(apa.authors || doc.apa_authors || '-')}">${escapeHtml(apa.authors || doc.apa_authors || '-')}</div>
                  </div>
                  <div class="col-span-2">
                     <div class="text-gray-500 mb-0.5">Publisher</div>
                     <div class="font-medium text-gray-800 truncate" title="${escapeHtml(apa.publisher || doc.apa_publisher || '-')}">${escapeHtml(apa.publisher || doc.apa_publisher || '-')}</div>
                  </div>
               </div>
            </div>
            
            <div>
               <h4 class="text-sm font-semibold text-gray-800 mb-2 flex justify-between items-end">
                  <span>Chunk Preview</span>
                  <span class="text-xs font-normal text-gray-500">Showing first ${chunks.length} of ${doc.chunk_count || 0} chunks</span>
               </h4>
               ${chunkHtml}
            </div>
         </div>
      </div>
      `;
      
      detailRow.querySelector('td > div').innerHTML = html;
  },
  
  async reingest(id) {
      if (!confirm('Are you sure you want to re-ingest this document? This will delete and recreate all its vector embeddings.')) return;
      
      try {
          const res = await fetch(`${API}/api/documents/${id}/reingest`, { method: 'POST' });
          if (!res.ok) {
              const err = await res.json();
              throw new Error(err.detail || `HTTP error ${res.status}`);
          }
          alert('Re-ingestion started successfully.');
          TabLibrary.load(); // reload current page
      } catch (e) {
          alert('Error re-ingesting document: ' + e.message);
      }
  },
  
  async deleteDoc(id, title) {
      if (!confirm(`Are you sure you want to DELETE "${title}" (ID: ${id})?\n\nThis will remove the document from the registry, delete all vector embeddings, and remove the file from MinIO. This action cannot be undone.`)) return;
      
      try {
          const res = await fetch(`${API}/api/documents/${id}`, { method: 'DELETE' });
          if (!res.ok) {
              const err = await res.json();
              throw new Error(err.detail || `HTTP error ${res.status}`);
          }
          alert('Document deleted successfully.');
          TabLibrary.load(); // reload current page
      } catch (e) {
          alert('Error deleting document: ' + e.message);
      }
  }
};

// ─── Tab: Upload ──────────────────────────────────────────────
const TabUpload = {
  pendingFile: null,
  
  initDropzone() {
      const dropzone = document.getElementById('upload-dropzone');
      const fileInput = document.getElementById('upload-file-input');
      
      dropzone.addEventListener('click', () => fileInput.click());
      
      dropzone.addEventListener('dragover', (e) => {
          e.preventDefault();
          dropzone.classList.add('border-green-500', 'bg-green-50');
      });
      
      dropzone.addEventListener('dragleave', () => {
          dropzone.classList.remove('border-green-500', 'bg-green-50');
      });
      
      dropzone.addEventListener('drop', (e) => {
          e.preventDefault();
          dropzone.classList.remove('border-green-500', 'bg-green-50');
          if (e.dataTransfer.files.length) {
              this.handleFileSelect(e.dataTransfer.files[0]);
          }
      });
      
      fileInput.addEventListener('change', (e) => {
          if (e.target.files.length) {
              this.handleFileSelect(e.target.files[0]);
          }
      });
  },
  
  async handleFileSelect(file) {
      this.pendingFile = file;
      document.getElementById('upload-dropzone').classList.add('hidden');
      document.getElementById('upload-file-info').classList.remove('hidden');
      document.getElementById('upload-success').classList.add('hidden');
      
      document.getElementById('upload-filename').textContent = file.name;
      document.getElementById('upload-filesize').textContent = fmtBytes(file.size);
      
      this.analyze(file);
  },
  
  async analyze(file) {
      const loading = document.getElementById('upload-analyze-loading');
      const errDiv = document.getElementById('upload-analyze-error');
      const formOverlay = document.getElementById('upload-form-overlay');
      const aiBadge = document.getElementById('upload-ai-badge');
      const submitBtn = document.getElementById('up-submit-btn');
      
      loading.classList.remove('hidden');
      errDiv.classList.add('hidden');
      formOverlay.classList.remove('hidden'); // Disable form
      aiBadge.classList.add('hidden');
      submitBtn.disabled = true;
      
      const formData = new FormData();
      formData.append('file', file);
      
      try {
          const res = await fetch(`${API}/api/documents/analyze-upload`, {
              method: 'POST',
              body: formData
          });
          
          if (!res.ok) {
              const err = await res.json();
              throw new Error(err.detail || `HTTP error ${res.status}`);
          }
          
          const data = await res.json();
          this.renderDraft(data.apa_draft || {}, file.name);
          
      } catch (e) {
          console.error(e);
          errDiv.textContent = `Analysis Error: ${e.message}\nYou can still upload by filling the metadata manually.`;
          errDiv.classList.remove('hidden');
          // Still allow manual upload
          this.renderDraft({}, file.name);
      } finally {
          loading.classList.add('hidden');
          formOverlay.classList.add('hidden'); // Enable form
          submitBtn.disabled = false;
      }
  },
  
  renderDraft(draft, filename) {
      document.getElementById('up-title').value = draft.title || filename.split('.').slice(0, -1).join('.') || filename;
      if (draft.topic) document.getElementById('up-topic').value = draft.topic;
      if (draft.type) document.getElementById('up-apa_type').value = draft.type;
      
      document.getElementById('up-apa_authors').value = draft.authors || '';
      document.getElementById('up-apa_year').value = draft.year || '';
      document.getElementById('up-apa_publisher').value = draft.publisher || '';
      
      // If AI drafted, show badge
      if (Object.keys(draft).length > 0) {
          document.getElementById('upload-ai-badge').classList.remove('hidden');
      }
  },
  
  async upload() {
      if (!this.pendingFile) return;
      
      const submitBtn = document.getElementById('up-submit-btn');
      const submitText = document.getElementById('up-submit-text');
      const spinner = document.getElementById('up-submit-spinner');
      const errDiv = document.getElementById('upload-analyze-error');
      
      submitBtn.disabled = true;
      submitText.textContent = 'Uploading & Ingesting...';
      spinner.classList.remove('hidden');
      errDiv.classList.add('hidden');
      
      const formData = new FormData();
      formData.append('file', this.pendingFile);
      formData.append('title', document.getElementById('up-title').value);
      formData.append('topic', document.getElementById('up-topic').value);
      formData.append('apa_type', document.getElementById('up-apa_type').value);
      formData.append('apa_authors', document.getElementById('up-apa_authors').value);
      formData.append('apa_year', document.getElementById('up-apa_year').value);
      formData.append('apa_publisher', document.getElementById('up-apa_publisher').value);
      
      try {
          const res = await fetch(`${API}/api/documents/upload`, {
              method: 'POST',
              body: formData
          });
          
          if (!res.ok) {
              const err = await res.json();
              throw new Error(err.detail || `HTTP error ${res.status}`);
          }
          
          const data = await res.json();
          this.showSuccess(data);
          
      } catch (e) {
          console.error(e);
          errDiv.textContent = `Upload Error: ${e.message}`;
          errDiv.classList.remove('hidden');
          submitBtn.disabled = false;
          submitText.textContent = 'Upload & Ingest to pgvector';
          spinner.classList.add('hidden');
      }
  },
  
  showSuccess(data) {
      document.getElementById('upload-form-overlay').classList.remove('hidden');
      
      const successDiv = document.getElementById('upload-success');
      const details = document.getElementById('upload-success-details');
      
      details.innerHTML = `
         <div><strong>ID:</strong> ${data.document_id}</div>
         <div><strong>Status:</strong> ${data.ingestion_status}</div>
         <div><strong>Chunks:</strong> ${data.chunks_ingested || 0}</div>
         <div><strong>Link:</strong> <a href="${data.source_link?.minio_url || '#'}" target="_blank" class="underline text-green-900">Open PDF</a></div>
         <div class="col-span-2 mt-2 pt-2 border-t border-green-200">
            <strong class="block mb-1">APA Citation:</strong>
            <div class="bg-white/50 p-2 rounded text-xs font-mono">${escapeHtml(data.apa_citation || 'None')}</div>
         </div>
      `;
      
      successDiv.classList.remove('hidden');
      
      // Update button state
      const submitBtn = document.getElementById('up-submit-btn');
      const submitText = document.getElementById('up-submit-text');
      const spinner = document.getElementById('up-submit-spinner');
      
      submitText.textContent = 'Upload Complete';
      spinner.classList.add('hidden');
      
      // Allow user to reset and upload another
      setTimeout(() => {
          submitBtn.disabled = false;
          submitText.textContent = 'Upload Another File';
          submitBtn.onclick = () => this.reset();
      }, 1000);
  },
  
  reset() {
      this.pendingFile = null;
      document.getElementById('upload-file-input').value = '';
      
      document.getElementById('upload-dropzone').classList.remove('hidden');
      document.getElementById('upload-file-info').classList.add('hidden');
      document.getElementById('upload-success').classList.add('hidden');
      document.getElementById('upload-analyze-error').classList.add('hidden');
      document.getElementById('upload-ai-badge').classList.add('hidden');
      
      document.getElementById('upload-form-overlay').classList.add('hidden');
      document.getElementById('upload-form').reset();
      
      const submitBtn = document.getElementById('up-submit-btn');
      const submitText = document.getElementById('up-submit-text');
      
      submitBtn.disabled = true;
      submitBtn.onclick = null; // Remove the 'Upload Another' override
      submitText.textContent = 'Upload & Ingest to pgvector';
  }
};

// ─── Tab: Evidence & Citations ────────────────────────────────
const TabEvidence = {
  async load(sessionId) {
      if (!sessionId) {
          sessionId = document.getElementById('ev-session-id').value.trim();
      } else {
          document.getElementById('ev-session-id').value = sessionId;
      }
      
      if (!sessionId) return;
      
      // Update URL hash without jumping
      window.history.replaceState(null, null, `#session=${sessionId}`);
      
      const loading = document.getElementById('ev-loading');
      const empty = document.getElementById('ev-empty');
      const content = document.getElementById('ev-content');
      
      loading.classList.remove('hidden');
      empty.classList.add('hidden');
      content.classList.add('hidden');
      
      try {
          // Fetch Evidence
          const resEv = await fetch(`${API}/api/evidence/session/${sessionId}`);
          if (!resEv.ok) throw new Error(`HTTP error! status: ${resEv.status}`);
          const evData = await resEv.json();
          const items = evData.evidence_items || (Array.isArray(evData) ? evData : []);
          
          if (items.length === 0) {
              loading.classList.add('hidden');
              empty.classList.remove('hidden');
              return;
          }
          
          // Fetch Coverage
          let coverageData = null;
          try {
              const resCov = await fetch(`${API}/api/evidence/session/${sessionId}/coverage`);
              if (resCov.ok) coverageData = await resCov.json();
          } catch(e) { console.warn('Coverage fetch failed:', e); }
          
          // Fetch Citations
          let citData = null;
          try {
              const resCit = await fetch(`${API}/api/citations/session/${sessionId}`);
              if (resCit.ok) citData = await resCit.json();
          } catch(e) { console.warn('Citations fetch failed:', e); }
          
          this.renderEvidence(items);
          if (coverageData) this.renderCoverage(coverageData);
          if (citData && citData.citations) this.renderCitations(citData.citations);
          
          loading.classList.add('hidden');
          content.classList.remove('hidden');
          
      } catch (e) {
          console.error(e);
          empty.innerHTML = `<div class="text-4xl mb-2">❌</div><p class="text-red-500 font-medium">Error loading session: ${escapeHtml(e.message)}</p>`;
          loading.classList.add('hidden');
          empty.classList.remove('hidden');
      }
  },
  
  renderEvidence(items) {
      const tbody = document.getElementById('ev-items-tbody');
      let html = '';
      
      items.forEach(ev => {
          let sourceHtml = escapeHtml(ev.source_ref || '-');
          if (ev.open_url) {
              sourceHtml = `<a href="${API}${ev.open_url}" target="_blank" class="text-indigo-600 hover:underline" title="Open source">${sourceHtml}</a>`;
          } else if (ev.original_url) {
              sourceHtml = `<a href="${escapeHtml(ev.original_url)}" target="_blank" class="text-indigo-600 hover:underline" title="Open original URL">${sourceHtml}</a>`;
          }
          
          html += `
          <tr class="hover:bg-gray-50">
             <td class="px-4 py-3 whitespace-nowrap text-gray-500">${escapeHtml(ev.evidence_id || '-')}</td>
             <td class="px-4 py-3">
                <div class="font-medium text-gray-800 text-xs mb-1 uppercase tracking-wider">${escapeHtml(ev.evidence_type || '-')}</div>
                <div class="truncate max-w-[200px] text-xs">${sourceHtml}</div>
             </td>
             <td class="px-4 py-3">
                <div class="text-xs text-gray-600 line-clamp-2 max-w-[300px]" title="${escapeHtml(ev.text_snippet || '-')}">${escapeHtml(ev.text_snippet || '-')}</div>
             </td>
             <td class="px-4 py-3 whitespace-nowrap">${trustedBadge(ev.trust_level || 'medium')}</td>
          </tr>
          `;
      });
      
      tbody.innerHTML = html;
  },
  
  renderCoverage(coverage) {
      const stats = document.getElementById('ev-coverage-stats');
      const bar = document.getElementById('ev-coverage-bar');
      const text = document.getElementById('ev-coverage-text');
      const list = document.getElementById('ev-claims-list');
      
      const score = Math.round((coverage.coverage_score || 0) * 100);
      
      stats.innerHTML = `
         <div class="bg-white p-3 rounded border border-gray-100 text-center">
            <div class="text-2xl font-bold text-gray-800">${coverage.total_claims || 0}</div>
            <div class="text-xs text-gray-500 uppercase tracking-wider">Total Claims</div>
         </div>
         <div class="bg-white p-3 rounded border border-green-100 text-center">
            <div class="text-2xl font-bold text-green-600">${coverage.supported_claims || 0}</div>
            <div class="text-xs text-green-600 uppercase tracking-wider">Supported</div>
         </div>
         <div class="bg-white p-3 rounded border border-yellow-100 text-center">
            <div class="text-2xl font-bold text-yellow-600">${coverage.partially_supported || 0}</div>
            <div class="text-xs text-yellow-600 uppercase tracking-wider">Partial</div>
         </div>
         <div class="bg-white p-3 rounded border border-red-100 text-center">
            <div class="text-2xl font-bold text-red-600">${coverage.unsupported_claims || 0}</div>
            <div class="text-xs text-red-600 uppercase tracking-wider">Unsupported</div>
         </div>
      `;
      
      // Determine bar colors based on score
      let colorClass = 'bg-green-500';
      if (score < 50) colorClass = 'bg-red-500';
      else if (score < 80) colorClass = 'bg-yellow-500';
      
      bar.innerHTML = `<div class="${colorClass} h-2.5 rounded-full" style="width: ${score}%"></div>`;
      text.textContent = `${score}% Supported`;
      
      // Claims list (if available in future API updates)
      if (coverage.claims && coverage.claims.length > 0) {
          let listHtml = '<ul class="space-y-2 mt-4">';
          coverage.claims.forEach(c => {
              const icon = c.support_level === 'supported' ? '✅' : (c.support_level === 'partially_supported' ? '⚠️' : '❌');
              listHtml += `<li class="flex items-start text-xs"><span class="mr-2">${icon}</span><span class="text-gray-700">${escapeHtml(c.claim_text)}</span></li>`;
          });
          listHtml += '</ul>';
          list.innerHTML = listHtml;
      } else {
          list.innerHTML = '<p class="text-xs text-gray-400 text-center italic mt-2">No detailed claim level data available.</p>';
      }
  },
  
  renderCitations(citations) {
      const list = document.getElementById('ev-citations-list');
      let html = '';
      
      citations.forEach(c => {
          let linkHtml = '';
          if (c.source_link && c.source_link.url) {
              const icon = c.source_link.icon === 'file-text' ? '📄' : (c.source_link.icon === 'database' ? '🗄️' : '🔗');
              linkHtml = `<a href="${API}${c.source_link.url}" target="_blank" class="mt-2 inline-flex items-center text-xs font-medium text-indigo-600 hover:text-indigo-800"><span class="mr-1">${icon}</span> ${escapeHtml(c.source_link.label || 'Open Link')}</a>`;
          } else if (c.open_url) {
              linkHtml = `<a href="${API}${c.open_url}" target="_blank" class="mt-2 inline-flex items-center text-xs font-medium text-indigo-600 hover:text-indigo-800"><span class="mr-1">🔗</span> Open Source</a>`;
          }
          
          html += `
          <div class="bg-white p-4 rounded-lg border border-gray-200 shadow-sm">
             <div class="flex justify-between items-start mb-2">
                <div class="flex items-center space-x-2">
                   <span class="font-mono text-xs bg-rose-50 text-rose-700 px-2 py-0.5 rounded border border-rose-200">${escapeHtml(c.citation_code)}</span>
                   ${trustedBadge(c.trust_level || 'medium')}
                   <span class="text-xs text-gray-500 uppercase tracking-wider">${escapeHtml(c.evidence_type || c.source_type || '-')}</span>
                </div>
                <div class="text-xs font-mono text-gray-400 bg-gray-100 px-1.5 py-0.5 rounded">${escapeHtml(c.evidence_id || '-')}</div>
             </div>
             
             <div class="text-sm text-gray-800 mb-2 pl-1 border-l-2 border-gray-200 ml-1">
                ${escapeHtml(c.inline_text || c.citation_text || '(Unknown)')}
             </div>
             
             <div class="text-xs text-gray-600 bg-gray-50 p-2 rounded border border-gray-100">
                ${c.reference_text || c.bibliography_text || 'No bibliography text available.'}
             </div>
             
             ${linkHtml}
          </div>
          `;
      });
      
      if (!html) {
          html = '<div class="text-sm text-gray-500 text-center py-4 bg-white border border-gray-200 rounded-lg">No citations generated for this session.</div>';
      }
      
      list.innerHTML = html;
  }
};

// ─── Init ─────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
    // Expose Tab functions to window for onclick handlers
    window.TabRAG = TabRAG;
    window.TabIndicators = TabIndicators;
    window.TabSQL = TabSQL;
    window.TabLibrary = TabLibrary;
    window.TabUpload = TabUpload;
    window.TabEvidence = TabEvidence;
    window.switchTab = switchTab;
    
    // Load library immediately when that tab is clicked, or initialize dropzone
    TabUpload.initDropzone();
});

