// static/app.js

const liveBtn        = document.getElementById('liveBtn');
const videoBtn       = document.getElementById('videoBtn');
const liveVideo      = document.getElementById('liveVideo');
const videoPlayer    = document.getElementById('videoPlayer');
const videoPlaceholder = document.getElementById('videoPlaceholder');
const fileInput      = document.getElementById('videoFile');
const detectVideoBtn = document.getElementById('detectVideoBtn');
const previewImg     = document.getElementById('previewImg');
const previewPlaceholder = document.getElementById('previewPlaceholder');
const logList        = document.getElementById('logList');

let liveInterval = null;
let currentMode  = null;      // 'live' 또는 'file'
const WINDOW_SIZE   = 45;
const POLL_INTERVAL = 200; // ms 단위(=0.2초)
const BUFFER_DELAY  = ((WINDOW_SIZE - 1) * (POLL_INTERVAL / 1000)).toFixed(2); 

let fallCount = 0;  // 누적된 낙상 탐지 횟수

// 캔버스: 필요한 시점에만 DOM에 붙이고 사용
const canvas = document.createElement('canvas');
const ctx    = canvas.getContext('2d');

function setMode(mode) {
  // 1) 기존 타이머 정리
  if (liveInterval) {
    clearInterval(liveInterval);
    liveInterval = null;
  }

  // 2) 서버 상태 리셋
  fetch('/api/live-reset', { method: 'POST' }).catch(console.warn);

  // 3) UI 초깃값: 모두 숨김
  liveVideo.classList.add('hidden');
  videoPlayer.classList.add('hidden');
  videoPlaceholder.classList.add('hidden');
  fileInput.classList.add('hidden');
  detectVideoBtn.classList.add('hidden');
  previewImg.style.display = 'none';
  previewPlaceholder.style.display = 'block';
  logList.replaceChildren(); 

  // 4) 버튼 스타일 토글
  liveBtn.classList.toggle('active', mode === 'live');
  videoBtn.classList.toggle('active', mode === 'video');

  async function startVideoDetection(file) {
  logList.replaceChildren(); 

  const form = new FormData();
  form.append('videoFile', file);
  const uploadRes = await fetch('/upload/video', { method: 'POST', body: form });
  if (!uploadRes.ok) { 
    alert('업로드 실패'); 
    return; 
  }


}

  
  if (mode === 'live') {
    // ───────────────── LIVE 모드 ─────────────────
    currentMode = 'live';
    liveVideo.classList.remove('hidden');

    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        liveVideo.srcObject = stream;
        liveVideo.addEventListener('loadedmetadata', () => {
          canvas.width  = liveVideo.videoWidth;
          canvas.height = liveVideo.videoHeight;
        });
        liveInterval = setInterval(captureFrame, POLL_INTERVAL);
      })
      .catch(err => console.error('웹캠 접근 실패:', err));

  } else if (mode === 'video') {
    // ───────────────── VIDEO 모드 ─────────────────
    currentMode = 'file';
    fileInput.classList.remove('hidden');
    detectVideoBtn.classList.remove('hidden');
    videoPlaceholder.classList.remove('hidden');

    // (1) 파일 선택 시 비디오 요소에 로드
    fileInput.onchange = async () => {
      const file = fileInput.files[0];
      if (!file) return;

      const url = URL.createObjectURL(file);
      videoPlayer.src = url;
      videoPlayer.classList.remove('hidden');
      videoPlaceholder.classList.add('hidden');

      // << 여기에 메타데이터 이벤트 등록 >>
      videoPlayer.addEventListener('loadedmetadata', () => {
        canvas.width  = videoPlayer.videoWidth;
        canvas.height = videoPlayer.videoHeight;
      }, { once: true });

      await startVideoDetection(file);
    };

    // (2) 검사 버튼 클릭 시 ‘실시간 탐지’ 시작


    detectVideoBtn.onclick = async () => {
      fallCount = 0;
      logList.replaceChildren();
      const file = fileInput.files[0];
      if (!file) return alert('파일 선택');

      const startTimeMs = Date.now();  // ← ⏱ 시작 시간 기록

      const eventSource = new EventSource('/detect/video');
      videoPlayer.play().catch(console.error);

      eventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        const time = parseFloat(data.time);
        const startTime = Math.max(0, time - 1.5);
        fallCount += 1;

        const li = document.createElement('li');
        li.textContent = `⚠ ${fallCount}번째 낙상 주의: ${startTime.toFixed(2)}s ~ ${time.toFixed(2)}s`;
        li.style.cursor = 'pointer';

        li.onclick = async () => {
          // ① 기존 선택 리셋
          logList.querySelectorAll('li').forEach(item =>
            item.classList.remove('selected-fall')
          );
          // ② 선택된 항목 강조
          li.classList.add('selected-fall');

          // ③ 평균 시점으로 이동하고 프레임 캡처
          const seekTime = (startTime + time) / 2;
          videoPlayer.currentTime = seekTime;
          videoPlayer.pause();
          await new Promise(res =>
            videoPlayer.addEventListener('seeked', res, { once: true })
          );

          const temp = document.createElement('canvas');
          temp.width = videoPlayer.videoWidth;
          temp.height = videoPlayer.videoHeight;
          const tctx = temp.getContext('2d');
          tctx.drawImage(videoPlayer, 0, 0, temp.width, temp.height);

          previewImg.src = temp.toDataURL('image/jpeg');
          previewImg.style.display = 'block';
          previewPlaceholder.style.display = 'none';
        };

        logList.appendChild(li);
      };

      eventSource.addEventListener("done", () => {
        eventSource.close();

        // 🕒 처리 시간 계산
        const elapsed = (Date.now() - startTimeMs) / 1000;

        const li = document.createElement('li');
        li.textContent = `✅ 낙상 분석 완료 (총 소요 시간: ${elapsed.toFixed(2)}초)`;
        li.style.fontWeight = 'bold';
        li.style.color = 'green';

        logList.appendChild(li);
      });
    };
  }
}

// 캔버스에 현재 프레임 그린 뒤, /api/live-detect로 POST
function captureFrame() {
  if (currentMode !== 'live') return;  // file 모드에선 동작 안 함
  let videoEl = null;
  if (currentMode === 'live') {
    videoEl = liveVideo;
  } else if (currentMode === 'file') {
    videoEl = videoPlayer;
  }
  if (!videoEl || videoEl.readyState < 2) return;

  ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
  canvas.toBlob(blob => {
    if (!blob) return;
    // (1) 현재 비디오 재생 시각도 쿼리로 보냄
    const timeParam = (currentMode === 'file')
      ? `?videoTime=${videoPlayer.currentTime.toFixed(2)}`
      : '';
    const url = `/api/live-detect${timeParam}`;
    fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/octet-stream' },
      body: blob
    })
    .then(res => res.json())
    .then(data => {
      if (Array.isArray(data) && data.length > 0) {
        updateLog(data);
      }
    })
    .catch(console.error);
  }, 'image/jpeg');
}

function stopCapture() {
  if (liveInterval) {
    clearInterval(liveInterval);
    liveInterval = null;
  }
}

function updateLog(items) {
  items.forEach(item => {
    let rawTime = item.timestamp; // 서버가 준 'videoTime' (예: "13.80") 또는 라이브 모드 시각
    let dispTime;

    if (currentMode === 'file') {
      // 비디오 모드인 경우: “실제 낙상 시점 ≒ serverTime − BUFFER_DELAY”
      // serverTime이 문자열 "13.80"이므로, 숫자로 바꾼 뒤 보정
      const detectedTime = parseFloat(rawTime);
      const delay = parseFloat(BUFFER_DELAY); // 8.80
      const approxFallTime = Math.max(0, detectedTime - delay);
      dispTime = approxFallTime.toFixed(2); // 예: 13.80 - 8.80 = 5.00 → "5.00"
    } else {
      // 라이브 모드일 때는 rawTime이 "HH:MM:SS" 형태이므로, 그대로 사용
      dispTime = rawTime;
    }

    const li = document.createElement('li');
    li.textContent = `[${item.timestamp}] — ${item.info}`;
    li.style.cursor = 'pointer';
    li.onclick = () => {
      previewImg.src = item.imageUrl;
      previewImg.style.display = 'block';
      previewPlaceholder.style.display = 'none';
    };
    logList.prepend(li);
  });
}

// 이벤트 연결 및 초기 모드 설정
liveBtn.addEventListener('click', () => setMode('live'));
videoBtn.addEventListener('click', () => setMode('video'));

// 페이지 로드 시 기본 모드: LIVE
setMode('live');
