$(document).ready(function() {
  // Toggle the mobile menu when the navbar burger is clicked
  $(".navbar-burger").click(function() {
    $(".navbar-burger").toggleClass("is-active");
    $(".navbar-menu").toggleClass("is-active");
  });
});
