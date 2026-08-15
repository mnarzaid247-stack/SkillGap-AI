const $ = (s,c=document)=>c.querySelector(s);
const $$ = (s,c=document)=>Array.from(c.querySelectorAll(s));

function clearElement(element){
  element.replaceChildren();
}

function createCheckIcon(){
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('fill', 'none');
  svg.setAttribute('stroke', 'currentColor');
  svg.setAttribute('stroke-width', '2.2');

  const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  path.setAttribute('d', 'M20 6L9 17l-5-5');

  svg.appendChild(path);
  return svg;
}

/* ---------- Jobs ---------- */
function renderJobs(jobs, onSelectJob){
  const currentJobs = jobs || [];

  const list = $('#jobList');
  clearElement(list);
  $('#jobCount').textContent = currentJobs.length;

  currentJobs.forEach((job, i)=>{
    const card = document.createElement('div');
    card.className = 'job-card';

    const company = job.company || 'غير مذكور';
    const location = job.location || 'غير مذكور';

    const jobMain = document.createElement('div');
    jobMain.className = 'job-main';

    const title = document.createElement('div');
    title.className = 'title';
    title.textContent = job.title || 'وظيفة بدون عنوان';

    const meta = document.createElement('div');
    meta.className = 'meta';

    const companyText = document.createTextNode(company + ' ');
    const separator = document.createElement('span');
    separator.textContent = '·';
    const locationText = document.createTextNode(' ' + location);

    meta.append(companyText, separator, locationText);
    jobMain.append(title, meta);

    const action = document.createElement('div');
    action.style.display = 'flex';
    action.style.alignItems = 'center';
    action.style.gap = '12px';

    const pick = document.createElement('span');
    pick.className = 'pick';
    pick.textContent = 'اختر للتحليل';

    action.appendChild(pick);
    card.append(jobMain, action);

    card.addEventListener('click', async ()=>{
      if(card.classList.contains('selected')) return;

      $$('.job-card').forEach(c=>{
        c.classList.remove('selected');
        const currentPick = c.querySelector('.pick');

        if(currentPick){
          currentPick.textContent = 'اختر للتحليل';
        }
      });

      card.classList.add('selected');
      pick.textContent = 'جارٍ تحليل الوظيفة…';

      await onSelectJob(i + 1, card);
    });

    list.appendChild(card);
  });
}

/* ---------- Requirements ---------- */
function renderReqs(requirements){
  const wrap = $('#reqChips');
  clearElement(wrap);

  const required = requirements?.required_skills || [];
  const preferred = requirements?.preferred_skills || [];
  const frameworks = requirements?.frameworks || [];

  const items = [
    ...required.map(x=>({label:x, type:'مطلوب'})),
    ...preferred.map(x=>({label:x, type:'مفضل'})),
    ...frameworks.map(x=>({label:x, type:'أداة/إطار'})),
  ];

  if(!items.length){
    const chip = document.createElement('div');
    chip.className = 'chip';
    chip.textContent = 'لم تُستخرج متطلبات كافية.';
    wrap.appendChild(chip);
    return;
  }

  items.forEach(item=>{
    const chip = document.createElement('div');
    chip.className = 'chip';
    chip.textContent = `${item.label} — ${item.type}`;
    wrap.appendChild(chip);
  });
}

/* ---------- Coverage & gaps ---------- */
function renderGaps(gap){
  const wrap = $('#gapList');
  clearElement(wrap);

  const coverage = Number(gap?.skill_coverage || 0);
  const matching = gap?.matching_required_skills || [];
  const missing = gap?.missing_required_skills || [];
  const evidence = gap?.evidence_gaps || [];

  const rows = [
    ...matching.map(skill=>({skill, pct:100, label:'متطابقة'})),
    ...missing.map(skill=>({skill, pct:0, label:'مفقودة'})),
    ...evidence.map(item=>({
      skill:item.skill || 'مهارة',
      pct:50,
      label:'تحتاج دليل'
    })),
  ];

  if(!rows.length){
    const row = document.createElement('div');
    row.className = 'gap-row';

    const skill = document.createElement('div');
    skill.className = 'skill';
    skill.textContent = 'لا توجد بيانات كافية';

    const empty = document.createElement('div');

    const pct = document.createElement('div');
    pct.className = 'pct';
    pct.textContent = '—';

    row.append(skill, empty, pct);
    wrap.appendChild(row);
  }

  rows.forEach(g=>{
    const row = document.createElement('div');
    row.className = 'gap-row';

    const cls = g.pct >= 75 ? 'hi' : g.pct >= 50 ? 'mid' : 'low';

    const skill = document.createElement('div');
    skill.className = 'skill';
    skill.textContent = g.skill;

    const label = document.createElement('div');
    label.style.fontSize = '11px';
    label.style.color = 'var(--text-faint)';
    label.style.fontWeight = '400';
    label.textContent = g.label;
    skill.appendChild(label);

    const barTrack = document.createElement('div');
    barTrack.className = 'bar-track';

    const barFill = document.createElement('div');
    barFill.className = `bar-fill ${cls}`;
    barFill.dataset.pct = String(g.pct);

    barTrack.appendChild(barFill);

    const pct = document.createElement('div');
    pct.className = 'pct';
    pct.textContent = `${g.pct}%`;

    row.append(skill, barTrack, pct);
    wrap.appendChild(row);
  });

  requestAnimationFrame(()=>{
    $$('.bar-fill').forEach(bar=>{
      bar.style.width = bar.dataset.pct + '%';
    });

    const circumference = 251;
    const safeCoverage = Math.max(0, Math.min(100, coverage));
    const offset = circumference - (safeCoverage / 100) * circumference;

    $('#ringFill').style.strokeDashoffset = offset;
    $('#ringNum').textContent = safeCoverage + '%';
    $('#ringFill').style.stroke =
      safeCoverage >= 75 ? 'var(--teal)' :
      safeCoverage >= 50 ? 'var(--blue)' :
      'var(--coral)';
  });

  const details = gap?.coverage_details || {};

  $('#coverageNote').textContent =
    `Skill Coverage: ${coverage}% — ${details.matched_required || 0} من ${details.total_required || 0} من المهارات المطلوبة متطابقة.`;
}

/* ---------- Career plan ---------- */
function renderPlan(recommendations){
  const wrap = $('#planList');
  clearElement(wrap);

  const gaps = recommendations?.priority_gaps || [];
  const learningOrder = recommendations?.learning_order || [];
  const project = recommendations?.portfolio_project || {};

  gaps.forEach(gap=>{
    const item = document.createElement('div');
    item.className = 'plan-item';

    const top = document.createElement('div');
    top.className = 'top';

    const title = document.createElement('h4');
    title.textContent = gap.skill || 'مهارة';

    const duration = document.createElement('span');
    duration.className = 'dur';
    duration.textContent = gap.priority || '';

    top.append(title, duration);

    const reason = document.createElement('p');
    reason.textContent = gap.reason || '';

    item.append(top, reason);
    wrap.appendChild(item);
  });

  if(learningOrder.length){
    const item = document.createElement('div');
    item.className = 'plan-item';

    const top = document.createElement('div');
    top.className = 'top';

    const title = document.createElement('h4');
    title.textContent = 'ترتيب التعلم المقترح';

    top.appendChild(title);

    const order = document.createElement('p');
    order.textContent = learningOrder.join(' ← ');

    item.append(top, order);
    wrap.appendChild(item);
  }

  if(project?.title){
    const techs = (project.technologies || []).join('، ');

    const item = document.createElement('div');
    item.className = 'plan-item';

    const top = document.createElement('div');
    top.className = 'top';

    const title = document.createElement('h4');
    title.textContent = project.title;

    const badge = document.createElement('span');
    badge.className = 'dur';
    badge.textContent = 'مشروع Portfolio';

    top.append(title, badge);

    const description = document.createElement('p');
    description.appendChild(
      document.createTextNode(project.description || '')
    );

    if(techs){
      description.appendChild(document.createElement('br'));

      const strong = document.createElement('strong');
      strong.textContent = 'التقنيات:';

      description.appendChild(strong);
      description.appendChild(document.createTextNode(' ' + techs));
    }

    item.append(top, description);
    wrap.appendChild(item);
  }

  if(!wrap.children.length){
    const item = document.createElement('div');
    item.className = 'plan-item';

    const top = document.createElement('div');
    top.className = 'top';

    const title = document.createElement('h4');
    title.textContent = 'لا توجد توصيات إضافية';

    top.appendChild(title);
    item.appendChild(top);
    wrap.appendChild(item);
  }
}

/* ---------- Summary ---------- */
function renderSummary(data){
  const wrap = $('#sumLog');
  clearElement(wrap);

  const recommendations = data.recommendations || {};
  const profile = data.candidate_profile || {};
  const selectedJob = data.selected_job || {};

  const lines = [
    selectedJob.title ? `تم تحليل الوظيفة: ${selectedJob.title}.` : null,
    profile.experience_level ? `مستوى الخبرة المستنتج: ${profile.experience_level}.` : null,
    recommendations.next_action ? `الخطوة التالية: ${recommendations.next_action}` : null,
    recommendations.apply_recommendation ? recommendations.apply_recommendation : null,
  ].filter(Boolean);

  if(!lines.length){
    lines.push('اكتمل التحليل بنجاح.');
  }

  lines.forEach(line=>{
    const row = document.createElement('div');
    row.className = 'sum-row';

    const text = document.createElement('span');
    text.textContent = line;

    row.append(createCheckIcon(), text);
    wrap.appendChild(row);
  });
}
