// scripts.js
document.addEventListener('DOMContentLoaded', function() {
    const menuButton = document.querySelector('.menu-button');
    const navMenu = document.querySelector('.nav-menu');

    menuButton.addEventListener('click', function() {
        navMenu.style.display = navMenu.style.display === 'block' ? 'none' : 'block';
    });

    // Fermer le menu lorsque l'on clique en dehors
    document.addEventListener('click', function(event) {
        if (!navMenu.contains(event.target) && !menuButton.contains(event.target)) {
            navMenu.style.display = 'none';
        }
    });
});
