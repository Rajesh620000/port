function scrollCarousel(direction) {
    const container = document.getElementById("skillsCarousel");
    const scrollAmount = 100;
    container.scrollBy({ left: direction * scrollAmount, behavior: 'smooth' });
}
  