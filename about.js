function openTab(event, tabId) {
  // Get all tab content elements and hide them
  const allTabs = document.querySelectorAll('.tab-content');
  allTabs.forEach(tab => tab.classList.remove('active'));

  // Remove 'active' class from all tab buttons
  const allButtons = document.querySelectorAll('.tab-link');
  allButtons.forEach(btn => btn.classList.remove('active'));

  // Show the clicked tab's content
  const selectedTab = document.getElementById(tabId);
  selectedTab.classList.add('active');

  // Mark the clicked button as active
  event.currentTarget.classList.add('active');
}

// gif trigger part
let isPopupOpen = false;
function handleGifClick() {
  const gif = document.getElementById("gifButton");
  const popup = document.getElementById("gifPopup");

  if (!isPopupOpen) {
      gif.src = "images/gif1.gif"; // Set your alternate GIF here
      popup.style.display = "block";
  } else {
      gif.src = "images/gif2.gif"; // Switch back to original
      popup.style.display = "none";
  }

  isPopupOpen = !isPopupOpen;
}