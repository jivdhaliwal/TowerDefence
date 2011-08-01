package towerdefence;

import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.newdawn.slick.*;
import org.newdawn.slick.font.effects.ColorEffect;
import org.newdawn.slick.geom.Vector2f;
import org.newdawn.slick.gui.*;
import org.newdawn.slick.particles.ParticleSystem;
import org.newdawn.slick.state.*;
import org.newdawn.slick.tiled.TiledMap;
import org.newdawn.slick.util.pathfinding.*;
import towerdefence.engine.*;
import towerdefence.engine.component.*;
import towerdefence.engine.entity.*;
import towerdefence.engine.levelLoader.*;
import towerdefence.engine.pathfinding.PathMap;

/**
 * TODO Refractor the hell out of this massive class
 * @author Jiv
 */
public class GameplayState extends BasicGameState implements ComponentListener {

    int stateID = 0;

    // Tower and Critter types
    public final static int NORMAL = 0;
    public final static int FIRE = 1;
    public final static int ICE = 2;
    // Critter type
    public final static int BOSS = 3;
    private Image tileHighlight;
    private Image validTile;
    private Image invalidTile;
    
    Tower selectedTower=null;
    

    private LevelLoader level;
    
    int critterCount;
    int tempCritterCount;

    // Update Time Counters
    int mouseCounter;
    int waveCounter;
    int generateCounter;

    boolean startWaves;

    // Map used to find critter paths
    public static PathMap pathmap;
    // Path finder used to search the pathmap
    private PathFinder finder;
    // Gives the last path found for the current unit
    private TiledMap map;
    
    
    // Map for gui underlay
    private Image guiBackground;
    private int guiLeftX = ((21*32) + 16);
    private int guiTopY = 48;
    private int guiBottomY = 656;

    CritterManager critterManager;
    private int waveNumber;
//    ArrayList<CritterManager> critterWaveList = new ArrayList<CritterManager>();
//    CritterManager critterWave;
    ArrayList<Critter> critterList;
    ArrayList<Tower> towerList = new ArrayList<Tower>();
    ArrayList<Tower> guiTowerList = new ArrayList<Tower>();

    Critter critter = null;
    ParticleSystem ps;

    private UnicodeFont unicodeFont;
    private UnicodeFont tempestaFont;
    private TowerManager towerFactory;
    private RenderWater waterAnimation;

    public static int TILESIZE;
    private int startX;
    private int startY;
    private int targetX;
    private int targetY;
    
    private int mouseY;
    private int mouseX;
    public static boolean towerSelected;
    
    private Image gameoverImage;
    private boolean gameOver;
    private Image levelCompleteImage;
    private boolean levelComplete;
    
    private GameContainer container;
    private TextField helpText;
    private boolean showHelp;
    private Image start;
    private MouseOverArea startArea;

	

    GameplayState() {
    }

    @Override
    public int getID() {
        return 1;
    }

    @Override
    public void enter(GameContainer container, StateBasedGame game) throws SlickException {
        init(container,game);
    }
    
    public void init(GameContainer container, StateBasedGame game) throws SlickException {
    	Player.getInstance().resetParams();
    	
    	validTile = ResourceManager.getInstance().getImage("VALID_TILE");
        invalidTile = ResourceManager.getInstance().getImage("INVALID_TILE");
        tileHighlight = validTile;
        
        gameoverImage = ResourceManager.getInstance().getImage("GAMEOVER");
        levelCompleteImage = ResourceManager.getInstance().getImage("LEVEL_COMPLETE");
    	
        gameOver=false;
        levelComplete=false;
        //loadLevel("data/levels/snake.xml");
//        guiMap = new TiledMap("data/gui/guiMap.tmx");
        guiBackground = ResourceManager.getInstance().getImage("GAME_GUI");

        map = new TiledMap(getLevel().getMapPath());

        // Get Start and Target positions from map
        startX = Integer.parseInt(map.getMapProperty("startX", null));
        startY = Integer.parseInt(map.getMapProperty("startY", null));
        targetX = Integer.parseInt(map.getMapProperty("targetX", null));
        targetY = Integer.parseInt(map.getMapProperty("targetY", null));

        TILESIZE=map.getTileWidth();
        
        pathmap = new PathMap(map);
        finder = new AStarPathFinder(pathmap, 500, false);
        
        setGuiTowers();
        
        towerFactory = new TowerManager();
        
        critterManager = new CritterManager(startX, startY,
                targetX, targetY, finder);
        waveNumber = 0;
        critterCount = getLevel().getWave(waveNumber).getNumCritters();
        
        waterAnimation = new RenderWater(map.getWidth()+5,map.getHeight());
        
        unicodeFont = new UnicodeFont("fonts/Jellyka_Estrya_Handwriting.ttf", 50, false, false);
        unicodeFont.getEffects().add(new ColorEffect(java.awt.Color.white));
        
        tempestaFont = new UnicodeFont("fonts/pf_tempesta_seven.ttf", 8, false, false);
        tempestaFont.getEffects().add(new ColorEffect(java.awt.Color.white));
        
        helpText = new TextField(container, container.getGraphics().getFont(), 
                (container.getWidth()/2-300), (container.getHeight()/2-80), 475, 200);
        helpText.setText("Game Help - F1 to toggle\n\nPress 1, 2 or 3 to select a tower\n"
                + "Left click to place a selected tower\n"
                + "Right click to cancel current selection\n"
                + "Mouse over a tower to see its info\n"
                + "Mouse over a tower and press delete to sell a tower\n"
                + "R : Restart\n"
                + "Esc : Level select screen\n"
                + "Manage your money and keep the critters at bay!");
        
        start = ResourceManager.getInstance().getImage("START_BUTTON");
        startArea = new MouseOverArea(container, start, guiLeftX-5, guiBottomY-75, 140, 40, this);
        startArea.setMouseOverColor(Color.black);
        startArea.setMouseDownColor(Color.black);
        
        startWaves = false;
        
        this.container = container;

    }

    public void loadLevel(String levelPath) throws SlickException {
        // Load Level
        setLevel(new LevelLoader(levelPath));
    }

    private void generateWaves() throws SlickException {
        // Reads level and generates the waves
        if(startWaves) {
            if (waveNumber < getLevel().getNumWaves()) {
                Wave currentWave = getLevel().getWave(waveNumber);
                if (waveCounter <= 0) {
                    if (generateCounter <= 0) {
                        if (critterCount > 0) {
                            critterManager.addCritter(String.valueOf((waveNumber * 1000) + critterCount),
                                    getLevel().getWave(waveNumber).getCritterType());
                            critterCount--;
                            generateCounter = currentWave.getTimeToSpawn();
                        } else if (critterCount <=0 ) {
                            waveCounter = currentWave.getTimeToWait();
                            waveNumber++;
                            if(waveNumber<getLevel().getNumWaves()){
                                critterCount = getLevel().getWave(waveNumber).getNumCritters();
                            }
                        }
                        
                    }
                } 
            } else if(critterManager.getCritters().size()==0){
            	levelComplete=true;
            }
        }
    }

    private void mouseListener(Input input) {
        input.addMouseListener(new MouseListener(){

            public void mouseWheelMoved(int change) {
            }

            public void mouseClicked(int button, int x, int y, int clickCount) {
                if (mouseCounter <= 0) {
                    int currentXTile = (int) Math.floor((x / GameplayState.TILESIZE));
                    int currentYTile = (int) Math.floor((y / GameplayState.TILESIZE));
                    if (button == 0) {
                        if (currentXTile <= 21 && selectedTower != null) {
                            if (pathmap.getTerrain(currentXTile, currentYTile) != PathMap.GRASS
                                    && pathmap.getTerrain(currentXTile, currentYTile) != PathMap.NOPLACE) {
                                if(Player.getInstance().getCash()-Player.getInstance().
                                        getTowerCost(selectedTower.getType()) >=0) {
//                                    selectedTower.setIsPlaced(true);
//                                    selectedTower.setActive(true);
//                                    selectedTower.setPosition(new Vector2f(currentXTile * TILESIZE, currentYTile * TILESIZE));
//                                    selectedTower.AddComponent(new LaserTowerComponent("LaserTower"));
//                                    towerFactory.addTower(selectedTower);
                                	try {
										towerFactory.addTower("selected", new Vector2f(currentXTile * TILESIZE, currentYTile * TILESIZE), 
												selectedTower.getType(), true, true);
									} catch (SlickException e) {
										// TODO Auto-generated catch block
										e.printStackTrace();
									}
                                    selectedTower = null;
                                    pathmap.setTowerTerrain(new Vector2f(currentXTile, currentYTile));
                                }

                                mouseCounter = 100;
                            }
                        } else if(currentXTile > 21) {
                            for(Tower guiTower : guiTowerList) {
                                if(currentXTile==guiTower.getTilePosition().x 
                                        && currentYTile==guiTower.getTilePosition().y) {
                                    try {
                                        setSelectedTower(guiTower.getType());
                                        mouseCounter=100;
                                    } catch (SlickException ex) {
                                        Logger.getLogger(GameplayState.class.getName()).log(Level.SEVERE, null, ex);
                                    }
                                }
                            }
                        }
                    }
                    if(button==1 && selectedTower!=null) {
                        selectedTower=null;
                    }
                }
            }

            public void mousePressed(int button, int x, int y) {
                
            }

            public void mouseReleased(int button, int x, int y) {
            }

            public void mouseMoved(int oldx, int oldy, int newx, int newy) {
                mouseX = newx;
                mouseY = newy;
                
                int currentXTile = (int) Math.floor((newx / GameplayState.TILESIZE));
                int currentYTile = (int) Math.floor((newy / GameplayState.TILESIZE));
                if (selectedTower != null) {
                    selectedTower.setPosition(new Vector2f(newx - 16, newy - 16));
                }

                if (mouseCounter <= 0 && currentXTile <= 21 && currentXTile >= 0 
                        && currentYTile >= 0 && currentYTile <= 21) {
                    
                    if (pathmap.getTerrain(currentXTile, currentYTile) != PathMap.GRASS
                            && pathmap.getTerrain(currentXTile, currentYTile) != PathMap.NOPLACE) {
                        tileHighlight = validTile;
                    } else {
                        tileHighlight = invalidTile;
                    }
                    mouseCounter = 32;
                } else if (currentXTile > 21) {
                    tileHighlight = null;
                }
            }

            public void mouseDragged(int oldx, int oldy, int newx, int newy) {
                
            }

            public void setInput(Input input) {
            }

            public boolean isAcceptingInput() {
                return true;
            }

            public void inputEnded() {
            }

            public void inputStarted() {
            }
        });
    }

    private void renderGuiText(Graphics g) {
        
        unicodeFont.drawString(guiLeftX+15, guiTopY+5,
                "Level: "+String.valueOf(getLevel().getLevelName()));
        
//        unicodeFont.addGlyphs("~!@!#!#$%___--");
//        
        unicodeFont.drawString(guiLeftX+15, guiTopY+65,
                "Cash : $"+String.valueOf(Player.getInstance().getCash()));
        
        unicodeFont.drawString(guiLeftX+15, guiTopY+125,
                "Health : "+String.valueOf(Player.getInstance().getHealth()));
        
//        unicodeFont.drawString(50, 110,
//                "# of Critters : " + String.valueOf(tempCritterCount), Color.white);
//        unicodeFont.drawString(50, 160,
//                "# of Towers : " + String.valueOf(towerFactory.getTowers().size()), Color.white);
        
        unicodeFont.drawString(guiLeftX+25, guiTopY+250, "TOWERS");
        
        if (startWaves && !gameOver) {
            unicodeFont.drawString(guiLeftX + 15, guiBottomY - 95,
                    " Next wave in:", Color.white);
            if (waveCounter > 0) {
                unicodeFont.drawString(guiLeftX + 30, guiBottomY - 70,
                        String.valueOf(waveCounter / 1000) + " seconds", Color.white);
            }
        } else {
//            unicodeFont.drawString(guiLeftX + 25, guiBottomY - 95,
//                    "Press Enter", Color.white);
//            unicodeFont.drawString(guiLeftX + 25, guiBottomY - 75,
//                    "   to begin", Color.white);
            startArea.render(container, g);
        }
        
        tempestaFont.drawString(guiLeftX+25, guiBottomY-115, "Press F1 for help");
    }
    
    private void setSelectedTower(int type) throws SlickException {
        selectedTower = new Tower("selected", false);
        selectedTower.setPosition(new Vector2f(mouseX-16,mouseY-16));
        selectedTower.setType(type);
        switch (type) {
			case Tower.NORMAL:
				selectedTower.AddComponent(new ImageRenderComponent("TowerRender",
						ResourceManager.getInstance().getImage("NORMAL_TOWER")));
				break;
			case Tower.FIRE:
				selectedTower.AddComponent(new ImageRenderComponent("TowerRender",
						ResourceManager.getInstance().getImage("FIRE_TOWER")));
				break;
			case Tower.ICE:
				selectedTower.AddComponent(new ImageRenderComponent("TowerRender",
						ResourceManager.getInstance().getImage("ICE_TOWER")));
				break;
		}
        selectedTower.setIsPlaced(false);
    }
    
    private void setGuiTowers() throws SlickException {
        guiTowerList.add(new Tower("normal", false));
        guiTowerList.add(new Tower("fire", false));
        guiTowerList.add(new Tower("ice", false));
        
        for(int i=0;i<3;i++) {
            guiTowerList.get(i).setPosition(new Vector2f( 
                    (float)(Math.floor((guiLeftX+16+(i*32)) / GameplayState.TILESIZE))*32, 
                    (float)(Math.floor((guiTopY+304)/GameplayState.TILESIZE))*32));
            guiTowerList.get(i).setType(i);
            switch (i) {
	    		case Tower.NORMAL:
	    			guiTowerList.get(i).AddComponent(new ImageRenderComponent("TowerRender",
	    					ResourceManager.getInstance().getImage("NORMAL_TOWER")));
	    			break;
	    		case Tower.FIRE:
	    			guiTowerList.get(i).AddComponent(new ImageRenderComponent("TowerRender",
	    					ResourceManager.getInstance().getImage("FIRE_TOWER")));
	    			break;
	    		case Tower.ICE:
	    			guiTowerList.get(i).AddComponent(new ImageRenderComponent("TowerRender",
	    					ResourceManager.getInstance().getImage("ICE_TOWER")));
	    			break;
    		}
            
            guiTowerList.get(i).setIsPlaced(true);
        }
        
    }

    public void update(GameContainer container, StateBasedGame game, int delta) throws SlickException {
        
    	Input input = container.getInput();
    	
    	if(input.isKeyDown(Input.KEY_F)) {
    		delta*=4;
    	}
    	
        mouseCounter -= delta;
        waveCounter -= delta;
        generateCounter -= delta;
        
        unicodeFont.loadGlyphs(100);
        tempestaFont.loadGlyphs(100);

//        ArrayList<Critter> tempCritterList = new ArrayList<Critter>();

        

        if (mouseCounter <= 0) {
            
            if(input.isKeyPressed(Input.KEY_F1)){showHelp = !showHelp;}
            
            if (input.isKeyPressed(Input.KEY_ESCAPE)) {
                game.enterState(TowerDefence.LEVELSELECTSTATE);
            }

            if (input.isKeyPressed(Input.KEY_ENTER)) {
                startWaves = true;
                mouseCounter=100;
            }
            
            if (input.isKeyPressed(Input.KEY_R)) {
                container.reinit();
            }
            
            if (input.isKeyPressed(Input.KEY_P)) {
                if(container.isPaused()) {
                    container.resume();
                } else {
                    container.pause();
                }
            }
            
            if (input.isKeyPressed(Input.KEY_1) || input.isKeyPressed(Input.KEY_NUMPAD1)) {
                setSelectedTower(Tower.NORMAL);
            }
            if (input.isKeyPressed(Input.KEY_2) || input.isKeyPressed(Input.KEY_NUMPAD2)) {
                setSelectedTower(Tower.FIRE);
            }
            if (input.isKeyPressed(Input.KEY_3) || input.isKeyPressed(Input.KEY_NUMPAD3
                      )) {
                setSelectedTower(Tower.ICE);;
            }
        }
        mouseListener(input);
        
        generateWaves();
         
        if(Player.getInstance().getHealth()>0) {
            critterManager.update(container, game, delta);
            if(critterManager.getCritters()!=towerFactory.getCritterList()) {
                towerFactory.updateCritterList(critterManager.getCritters());
            }

            towerFactory.update(container, game, delta);

            for(Tower guiTower : guiTowerList) {
                guiTower.update(container, game, delta);
            }

            if(selectedTower!=null) {
                towerSelected = true;
                selectedTower.update(container, game, delta);
            } else {
                towerSelected = false;
            }
        } else {
            gameOver=true;
        }

    }
    
    public void render(GameContainer container, StateBasedGame game, Graphics g) throws SlickException {
        g.setClip(TILESIZE, TILESIZE, TILESIZE*(map.getWidth()+3) , TILESIZE*(map.getHeight()-2));
        
        waterAnimation.render(container, game, g);
        
        map.render(0, 0,1);
        map.render(0, 0,2);
        
        if (tileHighlight != null) {
            tileHighlight.draw(((int) Math.floor((mouseX / GameplayState.TILESIZE))) * GameplayState.TILESIZE,
                    ((int) Math.floor((mouseY / GameplayState.TILESIZE))) * GameplayState.TILESIZE);
        }
        
        guiBackground.draw(21*32, 0);


//        for(CritterManager wave : critterWaveList) {
//            wave.render(container, game, g);
//            tempCritterCount+=wave.getCritters().size();
//        }

        if(!gameOver) {
            critterManager.render(container, game, g);
            tempCritterCount=critterManager.getCritters().size();

            towerFactory.render(container, game, g);
        } else {
            gameoverImage.drawCentered(container.getWidth()/2,container.getHeight()/2);
        }
        
        renderGuiText(g);

        for(Tower guiTower : guiTowerList) {
            guiTower.render(container, game, g);
        }
        
        if(selectedTower!=null) {
            selectedTower.render(container, game, g);
        }
        
        
        if(!gameOver && !levelComplete && showHelp){
            helpText.render(container, g);
        }
        
        if(levelComplete) {
        	levelCompleteImage.drawCentered(container.getWidth()/2,container.getHeight()/2);
        }

    }

    /**
     * @return the level
     */
    public LevelLoader getLevel() {
        return level;
    }

    /**
     * @param level the level to set
     */
    public void setLevel(LevelLoader level) {
        this.level = level;
    }

    @Override
    public void componentActivated(AbstractComponent source) {
        
        if(source==startArea) {
            startWaves=true;
        }
        
    }
}
