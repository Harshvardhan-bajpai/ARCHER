window.addEventListener('keydown', e => {
  const key = e.key;
  const k = key.toUpperCase();

  if (['W','A','S','D','G','H','X','F','L'].includes(k) || key === 'Tab') {
    fetch('/key', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ key })
    });
    e.preventDefault();
  }
});
