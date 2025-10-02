document.getElementById("uploadForm").addEventListener("submit", async (e) => {
  e.preventDefault();

  const fileInput = document.getElementById("resumeFile");
  const file = fileInput.files[0];
  
  if (!file) {
    alert("Please select a file first.");
    return;
  }

  // Show file info for debugging
  console.log("File name:", file.name);
  console.log("File size:", file.size, "bytes");
  console.log("File type:", file.type);

  const formData = new FormData();
  formData.append("file", file);

  // Show loading state
  document.getElementById("output").textContent = "Uploading and parsing resume...";

  try {
    const response = await fetch("http://127.0.0.1:8000/resume/parse-resume", {
      method: "POST",
      body: formData,
    });

    console.log("Response status:", response.status);

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Server error: ${errorText}`);
    }

    const data = await response.json();
    document.getElementById("output").textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    console.error("Error details:", err);
    document.getElementById("output").textContent = "Error: " + err.message;
  }
});