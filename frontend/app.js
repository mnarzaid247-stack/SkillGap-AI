(function(){
  let selectedFile = null;
  let currentThreadId = null;
  let currentJobs = [];

  /* ---------- Elements ---------- */
  const dropzone = $('#dropzone');
  const cvInput = $('#cvInput');
  const dzFileName = $('#dzFileName');
  const startBtn = $('#startBtn');
  const statusText = $('#statusText');
  const statusDot = $('#statusDot');
  const progressFill = $('#progressFill');
  const statusLine = $('#statusLine');

  /* ---------- Helpers ---------- */
  function setPhase(n, state){
    const p = document.querySelector(`.phase[data-phase="${n}"]`);

    if(!p) return;

    p.classList.remove('active', 'done');

    if(state){
      p.classList.add(state);
    }
  }

  function resetPhases(){
    $$('.phase').forEach(p=>{
      p.classList.remove('active', 'done');
    });

    progressFill.style.width = '0%';
  }

  function setLoading(isLoading){
    startBtn.disabled = isLoading || !selectedFile;

    startBtn.textContent = isLoading
      ? 'جارٍ التحليل…'
      : 'ابدأ التحليل';
  }

  function setError(message){
    statusText.textContent = 'حدث خطأ';
    statusLine.textContent = message;
    statusDot.style.background = 'var(--coral)';

    setLoading(false);
  }

  function resetCvErrorState(){
    dropzone.classList.remove('has-error');

    dzFileName.style.color = '';
    dzFileName.style.fontWeight = '';
  }

  function setCvError(message){
    selectedFile = null;
    currentThreadId = null;
    currentJobs = [];

    cvInput.value = '';

    dropzone.classList.remove('has-file');
    dropzone.classList.add('has-error');

    dzFileName.textContent = 'تعذر قراءة السيرة الذاتية';
    dzFileName.style.color = 'var(--coral)';
    dzFileName.style.fontWeight = '700';

    statusText.textContent = 'ملف السيرة الذاتية غير صالح';
    statusLine.textContent = message;
    statusDot.style.background = 'var(--coral)';

    startBtn.disabled = true;
    startBtn.textContent = 'ابدأ التحليل';
  }

  function isCvReadError(message){
    const text = String(message || '').toLowerCase();

    return (
      text.includes('تعذر قراءة السيرة الذاتية') ||
      text.includes('لا يحتوي على نص') ||
      text.includes('نص قابل للقراءة') ||
      text.includes('نص قابل للتحديد') ||
      text.includes('no readable text') ||
      text.includes('selectable text') ||
      text.includes('image-based') ||
      text.includes('scanned')
    );
  }

  async function parseError(response){
    try{
      const data = await response.json();

      return data.detail || 'حدث خطأ غير متوقع.';
    }catch(_){
      return 'حدث خطأ غير متوقع.';
    }
  }

  /* ---------- CV upload ---------- */
  dropzone.addEventListener('click', ()=>{
    cvInput.click();
  });

  dropzone.addEventListener('dragover', e=>{
    e.preventDefault();

    dropzone.classList.add('drag');
  });

  dropzone.addEventListener('dragleave', ()=>{
    dropzone.classList.remove('drag');
  });

  dropzone.addEventListener('drop', e=>{
    e.preventDefault();

    dropzone.classList.remove('drag');

    if(e.dataTransfer.files.length){
      handleFile(
        e.dataTransfer.files[0]
      );
    }
  });

  cvInput.addEventListener('change', e=>{
    if(e.target.files.length){
      handleFile(
        e.target.files[0]
      );
    }
  });

  function handleFile(file){
    resetCvErrorState();

    if(!file.name.toLowerCase().endsWith('.pdf')){
      setCvError(
        'النسخة الحالية تدعم ملفات PDF فقط.'
      );

      return;
    }

    if(file.size > 10 * 1024 * 1024){
      setCvError(
        'حجم ملف السيرة الذاتية يتجاوز 10MB.'
      );

      return;
    }

    selectedFile = file;

    dropzone.classList.add('has-file');

    dzFileName.textContent = file.name;

    statusText.textContent = 'تم اختيار السيرة الذاتية';
    statusLine.textContent = '';
    statusDot.style.background = 'var(--blue)';

    startBtn.disabled = false;
  }

  /* ---------- Start workflow ---------- */
  async function runAnalysis(){
    if(!selectedFile){
      setError(
        'اختاري ملف PDF أولًا.'
      );

      return;
    }

    const targetRole = $('#jobTitle')
      .value
      .trim();

    const location = $('#location')
      .value
      .trim();

    if(!targetRole || !location){
      setError(
        'اكتبي المسمى الوظيفي والموقع.'
      );

      return;
    }

    resetPhases();

    $('#results').classList.remove('show');

    clearElement($('#jobList'));
    clearElement($('#reqChips'));
    clearElement($('#gapList'));
    clearElement($('#planList'));
    clearElement($('#sumLog'));

    setLoading(true);

    statusText.textContent = 'التحليل قيد التنفيذ';
    statusDot.style.background = 'var(--blue)';

    setPhase(
      1,
      'active'
    );

    statusLine.textContent =
      'جاري تحليل السيرة الذاتية والبحث عن الوظائف بالتوازي…';

    progressFill.style.width = '15%';

    const form = new FormData();

    form.append(
      'cv',
      selectedFile
    );

    form.append(
      'target_role',
      targetRole
    );

    form.append(
      'location',
      location
    );

    try{
      const response = await fetch(
        '/api/start',
        {
          method: 'POST',
          body: form,
        }
      );

      if(!response.ok){
        const errorMessage =
          await parseError(response);

        if(isCvReadError(errorMessage)){
          setCvError(
            errorMessage
          );

          return;
        }

        throw new Error(
          errorMessage
        );
      }

      const data = await response.json();

      if(
        data.status !==
        'waiting_for_job_selection'
      ){
        throw new Error(
          'لم يصل النظام إلى مرحلة اختيار الوظيفة.'
        );
      }

      currentThreadId =
        data.thread_id;

      currentJobs =
        data.jobs || [];

      setPhase(
        1,
        'done'
      );

      progressFill.style.width = '25%';

      statusText.textContent =
        'اختر وظيفة للتحليل';

      statusLine.textContent =
        data.limited_results
          ? 'تم العثور على عدد محدود من الوظائف. اختر واحدة للمتابعة.'
          : 'تم العثور على وظائف حالية. اختر الوظيفة التي تريد تحليلها.';

      renderJobs(
        currentJobs,
        selectJob
      );

      $('#results')
        .classList
        .add('show');

      $('#results')
        .scrollIntoView({
          behavior: 'smooth',
          block: 'start',
        });

      startBtn.disabled = false;
      startBtn.textContent = 'إعادة البحث';

    }catch(error){
      setError(
        error.message ||
        'تعذر بدء التحليل.'
      );
    }
  }

  /* ---------- Resume workflow after selection ---------- */
  async function selectJob(
    selectedJobNumber,
    card
  ){
    if(!currentThreadId){
      setError(
        'جلسة التحليل غير موجودة. ابدئي التحليل من جديد.'
      );

      return;
    }

    try{
      setPhase(
        2,
        'active'
      );

      statusText.textContent =
        'تحليل الوظيفة المختارة';

      statusLine.textContent =
        'جاري استخراج متطلبات الوظيفة…';

      progressFill.style.width = '40%';

      const response = await fetch(
        '/api/select-job',
        {
          method: 'POST',

          headers: {
            'Content-Type':
              'application/json',
          },

          body: JSON.stringify({
            thread_id:
              currentThreadId,

            selected_job_index:
              selectedJobNumber,
          }),
        }
      );

      if(!response.ok){
        throw new Error(
          await parseError(response)
        );
      }

      const data =
        await response.json();

      if(
        data.status ===
        'waiting_for_job_selection'
      ){
        throw new Error(
          data.message ||
          'اختيار الوظيفة غير صالح.'
        );
      }

      setPhase(
        2,
        'done'
      );

      setPhase(
        3,
        'active'
      );

      progressFill.style.width = '65%';

      statusLine.textContent =
        'تم استخراج المتطلبات، وجاري حساب Skill Coverage…';

      renderReqs(
        data.job_requirements || {}
      );

      renderGaps(
        data.gap_analysis || {}
      );

      setPhase(
        3,
        'done'
      );

      setPhase(
        4,
        'active'
      );

      progressFill.style.width = '85%';

      statusLine.textContent =
        'جاري عرض توصيات Career Coach…';

      renderPlan(
        data.recommendations || {}
      );

      renderSummary(
        data
      );

      setPhase(
        4,
        'done'
      );

      progressFill.style.width = '100%';

      statusLine.textContent =
        'اكتمل التحليل.';

      statusText.textContent =
        'اكتمل التحليل بنجاح';

      statusDot.style.background =
        'var(--teal)';

      const pick =
        card.querySelector('.pick');

      if(pick){
        pick.textContent =
          'تم تحليل هذه الوظيفة';
      }

      $$('.job-card').forEach(c=>{
        if(c !== card){
          c.style.pointerEvents = 'none';
          c.style.opacity = '.65';
        }
      });

    }catch(error){
      card.classList.remove(
        'selected'
      );

      const pick =
        card.querySelector('.pick');

      if(pick){
        pick.textContent =
          'اختر للتحليل';
      }

      setError(
        error.message ||
        'تعذر تحليل الوظيفة المختارة.'
      );
    }
  }

  startBtn.addEventListener(
    'click',
    runAnalysis
  );
})();