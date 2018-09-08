// Return a C3 chart object
function getChart(elementId, classNames, probabilities) {
	classNames.unshift('class');
	probabilities.unshift('probability');

	return c3.generate({
		bindto: '#' + elementId,
		axis: {
			rotated: true,
			x: {
				type: 'category',
			},
			y: {
				max: 1,
				tick: {
					count: 11,
					format: function (d) {
						return (d * 100) + '%';
					},
					values: [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
				}
			}
		},
		legend: {
			show: false
		},
		data: {
			x: 'class',
			columns: [
				classNames,
				probabilities
			],
			types: {
				probability: 'bar'
			},
			color: function (inColor, data) {
				let colors = ["#FF0000",
					"#00FF00",
					"#0000FF",
					"#FFFF00",
					"#800080"];
				if (data.index !== undefined) {
					return colors[data.index % colors.length];
				}

				return inColor;
			}
		},
	})
}

function getPieChart(elementId, classNames, probabilities) {
	let pieDataArray = [];
	for (let i = 0; i < classNames.length; i++) {
		pieDataArray.push([classNames[i], probabilities[i]])
	}

	return c3.generate({
		bindto: '#' + elementId,
		data: {
			columns: pieDataArray,
			type: 'pie',
			onclick: function (d, i) {
				console.log("onclick", d, i);
			},
			onmouseover: function (d, i) {
				console.log("onmouseover", d, i);
			},
			onmouseout: function (d, i) {
				console.log("onmouseout", d, i);
			}
		}
	});
}
