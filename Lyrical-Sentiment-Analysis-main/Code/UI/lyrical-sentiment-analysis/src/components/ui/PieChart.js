import React from 'react';
import ApexCharts from 'react-apexcharts';

class PieChart extends React.Component {
	constructor(props) {
		super(props);
		console.log(props.results);
		this.state = {
			series: props.results.map(x => Math.round(x[1]*100000)/100000),
			options: {
				chart: {
					type: 'donut',
					background: '#181818'
				},
				labels: props.results.map(x => x[0]),
				colors: ['#008FFB', '#FD6A6A'],
				theme: {
					mode: 'dark'
				},
				stroke: {
					show: true,
					width: 1.5,
					colors: ['#fff']
				}
			}
		};
	}
	render() {
		return (
			<div style={{ display: 'flex', justifyContent: 'center' }}>
				<div id="chart" style={{ width: '400px'}}> 
					<ApexCharts options={this.state.options} series={this.state.series} type="donut" />
				</div>
				<div id="html-dist"></div>
			</div>
		);
	}
}

export default PieChart;