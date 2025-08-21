document.addEventListener("DOMContentLoaded", () => {
  const tryNowBtn = document.getElementById("try-now-btn");

  if (tryNowBtn) {
    tryNowBtn.addEventListener("click", () => {
      window.location.href = "verification.html"; // Redirect to text verification page
    });
  }
});
