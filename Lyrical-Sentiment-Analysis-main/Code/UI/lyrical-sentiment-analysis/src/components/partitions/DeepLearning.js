import { useContext, useRef } from "react";
import StateContext from "../../store/state-context";

import css from "./DeepLearning.module.css";
import PieChart from "../ui/PieChart";

function DeepLearning() {
	const stateContext = useContext(StateContext);

	let sortedResults = stateContext.results;
	sortedResults.sort((a, b) => {
		if (a[0] > b[0]) return -1;
		if (a[0] < b[0]) return 1;
		return 0;
	});
	
	return (
		<div>
			<div className={css.title}>Deep Learning Model Results</div>
			<br />
			<PieChart results={sortedResults}/>
		</div>
	);
}

export default DeepLearning;