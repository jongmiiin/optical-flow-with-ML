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
  logList.innerHTML = '';

  // 4) 버튼 스타일 토글
  liveBtn.classList.toggle('active', mode === 'live');
  videoBtn.classList.toggle('active', mode === 'video');

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
    fileInput.onchange = () => {
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
    };

    // (2) 검사 버튼 클릭 시 ‘실시간 탐지’ 시작
    detectVideoBtn.onclick = () => {
      const file = fileInput.files[0];
      if (!file) {
        alert('비디오 파일을 선택하세요.');
        return;
      }

      // (A) 비디오 재생
      // ─────────── 서버 상태를 매번 리셋 ───────────
      fetch('/api/live-reset', { method: 'POST' })
      .then(res => {
        if (!res.ok) throw new Error('라이브 리셋 실패');
        // (A) 상태 초기화 완료 후 비디오 재생
        return videoPlayer.play();
      })
      .catch(err => {
        console.error('비디오 모드 리셋 중 오류:', err);
        // 상태 리셋 실패해도 일단 비디오 재생 시도
        videoPlayer.play().catch(console.error);
      });

      // (B) 비디오 메타데이터 로드되면 캔버스 크기 설정
      // videoPlayer.addEventListener('loadedmetadata', () => {
      //   canvas.width  = videoPlayer.videoWidth;
      //   canvas.height = videoPlayer.videoHeight;
      // });

      // (C) 재생 시작 이벤트: 200ms마다 captureFrame 호출
      videoPlayer.addEventListener('play', () => {
        liveInterval = setInterval(captureFrame, POLL_INTERVAL);
      });

      // (D) 일시정지/종료 시 캡처 중단
      videoPlayer.addEventListener('pause', stopCapture);
      videoPlayer.addEventListener('ended', stopCapture);
    };
  }
}

// 캔버스에 현재 프레임 그린 뒤, /api/live-detect로 POST
function captureFrame() {
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
