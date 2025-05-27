// static/app.js
// 요소 참조
const liveBtn = document.getElementById('liveBtn');
const videoBtn = document.getElementById('videoBtn');
const liveVideo = document.getElementById('liveVideo');
const videoPlayer = document.getElementById('videoPlayer');
const videoPlaceholder = document.getElementById('videoPlaceholder');
const fileInput = document.getElementById('videoFile');
const detectVideoBtn = document.getElementById('detectVideoBtn');
const previewImg = document.getElementById('previewImg');
const previewPlaceholder = document.getElementById('previewPlaceholder');
const logList = document.getElementById('logList');

let liveInterval;

/**
 * 모드를 전환하며 관련 엘리먼트를 보여주거나 숨깁니다.
 */
function setMode(mode) {
  clearInterval(liveInterval);

  // UI 초기화
  liveVideo.classList.add('hidden');
  videoPlayer.classList.add('hidden');
  videoPlaceholder.classList.add('hidden');
  fileInput.classList.add('hidden');
  detectVideoBtn.classList.add('hidden');

  // 토글 버튼 스타일
  liveBtn.classList.toggle('active', mode === 'live');
  videoBtn.classList.toggle('active', mode === 'video');

  if (mode === 'live') {
    // LIVE 모드
    liveVideo.classList.remove('hidden');
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        liveVideo.srcObject = stream;
        liveInterval = setInterval(captureFrame, 1000);
      })
      .catch(err => console.error('웹캠 접근 실패:', err));
  } else {
    // VIDEO 모드
    fileInput.classList.remove('hidden');
    detectVideoBtn.classList.remove('hidden');
    videoPlaceholder.classList.remove('hidden');
  }
}

// 이벤트 연결
liveBtn.addEventListener('click', () => setMode('live'));
videoBtn.addEventListener('click', () => setMode('video'));

// 프레임 캡처 + 서버 전송 (LIVE)
function captureFrame() {
  const canvas = document.createElement('canvas');
  canvas.width = liveVideo.videoWidth;
  canvas.height = liveVideo.videoHeight;
  canvas.getContext('2d').drawImage(liveVideo, 0, 0);

  canvas.toBlob(blob => {
    fetch('/api/live-detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/octet-stream' },
      body: blob
    })
    .then(res => res.json())
    .then(updateLog)
    .catch(console.error);
  }, 'image/jpeg');
}

// VIDEO 모드: 파일 첨부 시 미리보기
fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (!file) return;
  videoPlayer.src = URL.createObjectURL(file);
  videoPlayer.classList.remove('hidden');
  videoPlaceholder.classList.add('hidden');
});

// VIDEO 모드: 검사 버튼 클릭
detectVideoBtn.addEventListener('click', () => {
  const file = fileInput.files[0];
  if (!file) return alert('비디오 파일을 선택하세요.');

  const form = new FormData();
  form.append('video', file);

  fetch('/api/video-detect', { method: 'POST', body: form })
    .then(res => res.json())
    .then(data => {
      console.log('[DEBUG] processedVideoUrl:', data.processedVideoUrl);
      
      // 로그 표시
      logList.innerHTML = '';
      (data.logs || []).forEach(item => {
        const li = document.createElement('li');
        li.textContent = `${item.timestamp} — ${item.info}`;
        li.onclick = () => {
          previewImg.src = item.imageUrl;
          previewImg.style.display = 'block';
          previewPlaceholder.style.display = 'none';
        };
        logList.append(li);
      });

      // 처리된 비디오 재생
      videoPlayer.src = data.processedVideoUrl;
      videoPlayer.load();
      videoPlayer.play().catch(console.error);
    })
    .catch(console.error);
});

// 로그 업데이트 (LIVE + 기본)
function updateLog(data) {
  logList.innerHTML = '';
  data.forEach(item => {
    const li = document.createElement('li');
    li.textContent = `${item.timestamp} — ${item.info}`;
    li.onclick = () => {
      previewImg.src = item.imageUrl;
      previewImg.style.display = 'block';
      previewPlaceholder.style.display = 'none';
    };
    logList.prepend(li);
  });
}

// 초기 모드 설정
setMode('live');
