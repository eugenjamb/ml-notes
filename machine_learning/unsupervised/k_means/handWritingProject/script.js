var canvasIds = ['myCanvas', 'myCanvas2', 'myCanvas3', 'myCanvas4'];
var contexts = {};
var drawState = {};
var STORAGE_KEY = 'digitDrawings';
var DIGITS_MAX_VALUE = 16;

function InitThis() {
    canvasIds.forEach(function (canvasId) {
        var canvas = document.getElementById(canvasId);
        var context = canvas.getContext('2d');

        contexts[canvasId] = context;
        drawState[canvasId] = {
            mousePressed: false,
            lastX: 0,
            lastY: 0
        };

        canvas.addEventListener('mousedown', function (e) {
            var point = getCanvasPoint(this, e);
            drawState[canvasId].mousePressed = true;
            drawState[canvasId].lastX = point.x;
            drawState[canvasId].lastY = point.y;
        });

        canvas.addEventListener('mousemove', function (e) {
            if (!drawState[canvasId].mousePressed) {
                return;
            }

            var point = getCanvasPoint(this, e);
            drawLine(canvasId, point.x, point.y);
        });

        canvas.addEventListener('mouseup', function () {
            drawState[canvasId].mousePressed = false;
        });

        canvas.addEventListener('mouseleave', function () {
            drawState[canvasId].mousePressed = false;
        });
    });
}

function getCanvasPoint(canvas, event) {
    var rect = canvas.getBoundingClientRect();
    return {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top
    };
}

function drawLine(canvasId, x, y) {
    var context = contexts[canvasId];
    var state = drawState[canvasId];
    var widthSelector = document.getElementById('selWidth');
    var colorSelector = document.getElementById('selColor');

    context.beginPath();
    context.strokeStyle = colorSelector.value;
    context.lineWidth = widthSelector.value;
    context.lineJoin = 'round';
    context.lineCap = 'round';
    context.moveTo(state.lastX, state.lastY);
    context.lineTo(x, y);
    context.stroke();
    context.closePath();

    state.lastX = x;
    state.lastY = y;
}

function clearArea() {
    canvasIds.forEach(function (canvasId) {
        var context = contexts[canvasId];
        context.setTransform(1, 0, 0, 1, 0, 0);
        context.clearRect(0, 0, context.canvas.width, context.canvas.height);
    });

    document.getElementById('display').textContent = '';
    document.getElementById('display2').textContent = '';
    document.getElementById('display3').textContent = '';
    document.getElementById('display4').textContent = '';
    document.getElementById('opening_bracket').textContent = '';
    document.getElementById('closing_bracket').textContent = '';
    document.getElementById('storage_status').textContent = '';
}

function array() {
    var drawings = canvasIds.map(function (canvasId) {
        return extractCanvasArray(contexts[canvasId]);
    });

    document.getElementById('opening_bracket').textContent = '[';
    document.getElementById('display').textContent = JSON.stringify(drawings[0]) + ',';
    document.getElementById('display2').textContent = JSON.stringify(drawings[1]) + ',';
    document.getElementById('display3').textContent = JSON.stringify(drawings[2]) + ',';
    document.getElementById('display4').textContent = JSON.stringify(drawings[3]);
    document.getElementById('closing_bracket').textContent = ']';

    localStorage.setItem(STORAGE_KEY, JSON.stringify(drawings));
    window.latestDigitDrawings = drawings;

    document.getElementById('storage_status').textContent =
        'Saved ' + drawings.length + ' model-format arrays to localStorage key "' + STORAGE_KEY + '".';
}

function extractCanvasArray(context) {
    var width = context.canvas.width;
    var height = context.canvas.height;
    var imageData = context.getImageData(0, 0, width, height).data;
    var blockSize = 10;
    var gridWidth = width / blockSize;
    var values = [];

    for (var blockY = 0; blockY < gridWidth; blockY++) {
        for (var blockX = 0; blockX < gridWidth; blockX++) {
            var totalDarkness = 0;

            for (var y = 0; y < blockSize; y++) {
                for (var x = 0; x < blockSize; x++) {
                    var pixelX = blockX * blockSize + x;
                    var pixelY = blockY * blockSize + y;
                    var pixelIndex = (pixelY * width + pixelX) * 4;
                    var red = imageData[pixelIndex];
                    var green = imageData[pixelIndex + 1];
                    var blue = imageData[pixelIndex + 2];
                    var alpha = imageData[pixelIndex + 3];
                    var grayscale = (red + green + blue) / 3;
                    var darkness = alpha === 0 ? 0 : 1 - grayscale / 255;

                    totalDarkness += darkness;
                }
            }

            values.push(Number(((totalDarkness / (blockSize * blockSize)) * DIGITS_MAX_VALUE).toFixed(2)));
        }
    }

    return values;
}
