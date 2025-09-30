const yearEl = document.getElementById('year');
if (yearEl) yearEl.textContent = String(new Date().getFullYear());

const grid = document.getElementById('demo-grid');
const filterButtons = Array.from(document.querySelectorAll('.chip'));
let demos = [];
let activeFilter = 'all';

function renderGrid() {
  if (!grid) return;
  const items = demos.filter(d => {
    if (activeFilter === 'all') return true;
    return d.tags && d.tags.includes(activeFilter);
  });
  grid.innerHTML = items.map(item => {
    const tags = (item.tags || []).map(t => `<span class="tag">#${t}</span>`).join('');
    const filename = item.file || '';
    const lower = filename.toLowerCase();
    const isVideo = lower.endsWith('.mp4') || lower.endsWith('.webm') || item.type === 'video';
    // Allow optional 'dir' to override default directories. Fall back by extension.
    const dir = item.dir || (isVideo ? 'demos/video' : 'demos/audio');
    const src = `${dir}/${filename}`;
    const mediaTag = isVideo
      ? `<video controls preload="metadata" src="${src}" style="width:100%; border-radius:10px; background:#000"></video>`
      : `<audio controls preload="none" src="${src}"></audio>`;
    return `
      <article class="card" role="region" aria-label="${item.title}">
        <div class="card-body">
          <h3 class="card-title">${item.title}</h3>
          <div class="card-tags">${tags}</div>
          ${mediaTag}
        </div>
      </article>
    `;
  }).join('');
}

filterButtons.forEach(btn => {
  btn.addEventListener('click', () => {
    filterButtons.forEach(b => b.classList.remove('is-active'));
    btn.classList.add('is-active');
    activeFilter = btn.dataset.filter || 'all';
    renderGrid();
  });
});

fetch('demos/demo_config.json')
  .then(res => res.ok ? res.json() : [])
  .then(json => { demos = json; renderGrid(); })
  .catch(() => { demos = []; renderGrid(); });


const demoGrid = document.getElementById('demo-grid');

// const demoGrid = document.getElementById('demo-grid');
const prevBtn = document.querySelector('.carousel-btn.prev');
const nextBtn = document.querySelector('.carousel-btn.next');

const cardWidth = 256; // 每个卡片宽度 + gap，大约=240+16

prevBtn.addEventListener('click', () => {
  demoGrid.scrollBy({ left: -cardWidth * 3, behavior: 'smooth' });
});

nextBtn.addEventListener('click', () => {
  demoGrid.scrollBy({ left: cardWidth * 3, behavior: 'smooth' });
});

