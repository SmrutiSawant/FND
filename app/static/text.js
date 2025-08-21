document.addEventListener("DOMContentLoaded", () => {
  const verifyBtn = document.getElementById("verify-btn");
  const textInput = document.getElementById("input-text");
  const fakeBtn = document.getElementById("fake-result");
  const realBtn = document.getElementById("real-result");

  verifyBtn.addEventListener("click", () => {
    const userText = textInput.value.trim();

    if (!userText) {
      alert("Please enter some text to verify.");
      return;
    }

    // Dummy classification logic (replace with actual API later)
    const isFake = userText.toLowerCase().includes("fake");

    if (isFake) {
      fakeBtn.style.display = "inline-block";
      realBtn.style.display = "none";
    } else {
      fakeBtn.style.display = "none";
      realBtn.style.display = "inline-block";
    }
  });
});
