import { createContext, useState } from "react";

const StateContext = createContext({
	lyrics: "",
	model: "",
	isRunning: false,
	results: null,
	updateModel: (newData) => {},
});

export function StateContextProvider(props) {
	const [data, setData] = useState({
		lyrics: "",
		model: "",
		isRunning: false,
		results: null
	});
	
	function updateModelHandler(newData) {
		setData(previousState => {
			return {
				...previousState,
				...newData
			}
		});
	}
	
	const context = {
		lyrics: data.lyrics,
		model: data.model,
		isRunning: data.isRunning,
		results: data.results,
		updateModel: updateModelHandler,
	}
	
	return <StateContext.Provider value={context}>{props.children}</StateContext.Provider>;
}

export default StateContext;